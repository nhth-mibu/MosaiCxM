//! Incremental library indexing.
//!
//! Change detection: a file is "unchanged" when its relative path, size and
//! modification time (ns) match the catalogue and a descriptor of the current
//! version exists for its content hash. Otherwise the file is read and hashed
//! (BLAKE3). If a current descriptor already exists for that hash (e.g. a
//! moved, copied or touched file, or a duplicate), it is reused without
//! decoding. Only new content is decoded.
//!
//! Limitations: an edit that preserves both size and mtime is not detected
//! unless `verify` is set (re-hashes everything). A missing source folder is
//! reported and its records are kept (it may be an unplugged drive).

use crate::catalog::{path_str, Catalog, Hash, STATUS_OK, STATUS_SKIPPED};
use crate::decode;
use crate::descriptor::{self, ANALYSIS_SHORT_SIDE, THUMB_SHORT_SIDE};
use crate::job::Job;
use anyhow::Result;
use rayon::prelude::*;
use rusqlite::params;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "jpe", "jfif", "png", "webp"];

#[derive(Debug, Clone)]
pub struct IndexOptions {
    pub threads: usize,
    pub verify: bool,
    pub batch: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self { threads: default_threads(), verify: false, batch: 128 }
    }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)
}

#[derive(Debug, Clone, Default)]
pub struct IndexReport {
    pub roots_missing: Vec<PathBuf>,
    pub files_seen: u64,
    pub added: u64,
    pub changed: u64,
    pub removed: u64,
    pub unchanged: u64,
    pub decoded: u64,
    pub reused_descriptors: u64,
    pub skipped: Vec<(PathBuf, String)>,
    pub ignored_other_files: u64,
    pub walk_errors: Vec<String>,
    pub duplicate_files: u64,
    pub elapsed_ms: u128,
    pub walk_ms: u128,
    pub cancelled: bool,
}

struct WorkItem {
    root_id: i64,
    abs: PathBuf,
    rel: String,
    size: i64,
    mtime_ns: i64,
    existing: bool,
}

enum WorkResult {
    Ok { hash: Hash, attribution: (Option<String>, Option<String>), analysis: Option<Analysis> },
    Skipped(String),
}

struct Analysis {
    desc: descriptor::Descriptor,
    thumb: Vec<u8>,
    has_icc: bool,
    had_alpha: bool,
}

pub fn update_library(cat: &Catalog, lib: i64, opts: &IndexOptions, job: &Job) -> Result<IndexReport> {
    let t0 = Instant::now();
    let mut rep = IndexReport::default();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.threads.max(1)).build()?;
    let mut known: HashSet<Hash> = {
        let mut st = cat.conn.prepare("SELECT content_hash FROM descriptor WHERE version = ?1")?;
        let v = st
            .query_map([descriptor::DESCRIPTOR_VERSION], |r| r.get::<_, Vec<u8>>(0))?
            .filter_map(|r| r.ok().and_then(|v| v.try_into().ok()))
            .collect();
        v
    };
    for root in cat.roots(lib)? {
        job.check()?;
        if !root.path.is_dir() {
            rep.roots_missing.push(root.path.clone());
            continue;
        }
        let tw = Instant::now();
        job.report("scanning folders", 0, 0);
        let (seen, walk_ok) = walk_root(&root.path, &mut rep);
        rep.walk_ms += tw.elapsed().as_millis();
        rep.files_seen += seen.len() as u64;
        let existing = cat.files_of_root(root.id)?;
        let mut work = Vec::new();
        for (rel, (size, mtime)) in &seen {
            match existing.get(rel) {
                Some(e) if !opts.verify && e.size == *size && e.mtime_ns == *mtime => {
                    let fresh = e.status == STATUS_SKIPPED || e.hash.map(|h| known.contains(&h)).unwrap_or(false);
                    if fresh {
                        rep.unchanged += 1;
                        continue;
                    }
                    work.push(WorkItem { root_id: root.id, abs: root.path.join(rel), rel: rel.clone(), size: *size, mtime_ns: *mtime, existing: true });
                }
                e => work.push(WorkItem {
                    root_id: root.id,
                    abs: root.path.join(rel),
                    rel: rel.clone(),
                    size: *size,
                    mtime_ns: *mtime,
                    existing: e.is_some(),
                }),
            }
        }
        let total = work.len() as u64;
        let mut done = 0u64;
        for chunk in work.chunks(opts.batch.max(1)) {
            if job.is_cancelled() {
                rep.cancelled = true;
                break;
            }
            let known_ref = &known;
            let results: Vec<WorkResult> = pool.install(|| chunk.par_iter().map(|w| process_file(&w.abs, known_ref)).collect());
            let tx = cat.conn.unchecked_transaction()?;
            for (w, r) in chunk.iter().zip(results) {
                if w.existing {
                    rep.changed += 1;
                } else {
                    rep.added += 1;
                }
                match r {
                    WorkResult::Ok { hash, attribution, analysis } => {
                        if let Some(a) = analysis {
                            cat.put_descriptor(&hash, &a.desc, a.has_icc, a.had_alpha, &a.thumb)?;
                            rep.decoded += 1;
                        } else {
                            rep.reused_descriptors += 1;
                        }
                        known.insert(hash);
                        upsert(cat, w, Some(&hash), STATUS_OK, None, &attribution)?;
                    }
                    WorkResult::Skipped(reason) => {
                        rep.skipped.push((w.abs.clone(), reason.clone()));
                        upsert(cat, w, None, STATUS_SKIPPED, Some(&reason), &(None, None))?;
                    }
                }
            }
            tx.commit()?;
            done += chunk.len() as u64;
            job.report("indexing photos", done, total);
        }
        if rep.cancelled {
            break;
        }
        if walk_ok {
            let tx = cat.conn.unchecked_transaction()?;
            for (rel, e) in &existing {
                if !seen.contains_key(rel) {
                    cat.conn.execute("DELETE FROM file WHERE id = ?1", [e.id])?;
                    rep.removed += 1;
                }
            }
            tx.commit()?;
        }
        apply_attribution_sidecar(cat, root.id, &root.path)?;
    }
    rep.duplicate_files = cat.conn.query_row(
        "SELECT COUNT(*) - COUNT(DISTINCT f.content_hash) FROM file f JOIN root r ON f.root_id = r.id
         WHERE r.library_id = ?1 AND f.status = 0",
        [lib],
        |r| r.get::<_, i64>(0),
    )? as u64;
    rep.elapsed_ms = t0.elapsed().as_millis();
    Ok(rep)
}

fn upsert(cat: &Catalog, w: &WorkItem, hash: Option<&Hash>, status: i64, err: Option<&str>, attr: &(Option<String>, Option<String>)) -> Result<()> {
    cat.conn.execute(
        "INSERT INTO file(root_id, rel_path, size, mtime_ns, content_hash, status, error, artist, copyright)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(root_id, rel_path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns,
           content_hash = excluded.content_hash, status = excluded.status, error = excluded.error,
           artist = excluded.artist, copyright = excluded.copyright",
        params![w.root_id, w.rel, w.size, w.mtime_ns, hash.map(|h| h.to_vec()), status, err, attr.0, attr.1],
    )?;
    Ok(())
}

fn walk_root(root: &Path, rep: &mut IndexReport) -> (HashMap<String, (i64, i64)>, bool) {
    let mut seen = HashMap::new();
    let mut ok = true;
    let it = walkdir::WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| {
        e.depth() == 0 || !e.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
    });
    for entry in it {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                ok = false;
                rep.walk_errors.push(e.to_string());
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry.path().extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
        if !ext.map(|e| IMAGE_EXTENSIONS.contains(&e.as_str())).unwrap_or(false) {
            rep.ignored_other_files += 1;
            continue;
        }
        let rel = match entry.path().strip_prefix(root).ok().and_then(|p| p.to_str()) {
            Some(r) => r.to_string(),
            None => {
                rep.skipped.push((entry.path().to_path_buf(), "path is not valid UTF-8".into()));
                continue;
            }
        };
        match entry.metadata() {
            Ok(m) => {
                let mtime = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                seen.insert(rel, (m.len() as i64, mtime));
            }
            Err(e) => {
                ok = false;
                rep.walk_errors.push(e.to_string());
            }
        }
    }
    (seen, ok)
}

fn process_file(path: &Path, known: &HashSet<Hash>) -> WorkResult {
    let bytes = match std::fs::metadata(path) {
        Ok(m) if m.len() > decode::MAX_FILE_BYTES => return WorkResult::Skipped(format!("file too large ({} bytes)", m.len())),
        Ok(_) => match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => return WorkResult::Skipped(format!("cannot read: {e}")),
        },
        Err(e) => return WorkResult::Skipped(format!("cannot stat: {e}")),
    };
    if decode::sniff(&bytes).is_none() {
        return WorkResult::Skipped("unsupported or unrecognised format".into());
    }
    let hash: Hash = *blake3::hash(&bytes).as_bytes();
    let attribution = exif_attribution(&bytes);
    if known.contains(&hash) {
        return WorkResult::Ok { hash, attribution, analysis: None };
    }
    match analyse(&bytes) {
        Ok(a) => WorkResult::Ok { hash, attribution, analysis: Some(a) },
        Err(e) => WorkResult::Skipped(e.to_string()),
    }
}

fn analyse(bytes: &[u8]) -> Result<Analysis> {
    let d = decode::decode_bytes(bytes, Some(ANALYSIS_SHORT_SIDE))?;
    let (w, h) = d.image.dimensions();
    if w < 2 || h < 2 {
        anyhow::bail!("image too small ({w}x{h})");
    }
    let desc = descriptor::compute(&d.image, d.info.full_width, d.info.full_height);
    let s = THUMB_SHORT_SIDE as f64 / w.min(h) as f64;
    let (tw, th) = if s < 1.0 { (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1)) } else { (w, h) };
    let thumb_img = if (tw, th) == (w, h) { d.image.clone() } else { decode::resize_crop(&d.image, [0.0, 0.0, 1.0, 1.0], tw, th)? };
    let thumb = decode::encode_jpeg(&thumb_img, 85)?;
    Ok(Analysis { desc, thumb, has_icc: d.info.has_icc, had_alpha: d.info.had_alpha })
}

fn exif_attribution(bytes: &[u8]) -> (Option<String>, Option<String>) {
    let Ok(ex) = exif::Reader::new().read_from_container(&mut std::io::Cursor::new(bytes)) else { return (None, None) };
    let get = |tag| {
        ex.get_field(tag, exif::In::PRIMARY).and_then(|f| match &f.value {
            exif::Value::Ascii(v) => {
                let s = v.iter().map(|b| String::from_utf8_lossy(b).trim().to_string()).collect::<Vec<_>>().join(" ");
                let s = s.trim().to_string();
                (!s.is_empty()).then_some(s)
            }
            _ => None,
        })
    };
    (get(exif::Tag::Artist), get(exif::Tag::Copyright))
}

/// Optional `attribution.csv` in a source folder with a header row containing
/// `path` and any of `creator`/`artist`, `copyright`, `license`, `source`/`url`.
/// Values override EXIF fields for the listed files.
fn apply_attribution_sidecar(cat: &Catalog, root_id: i64, root: &Path) -> Result<()> {
    let p = root.join("attribution.csv");
    let Ok(text) = std::fs::read_to_string(&p) else { return Ok(()) };
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else { return Ok(()) };
    let cols: Vec<String> = parse_csv_line(header).into_iter().map(|c| c.trim().to_ascii_lowercase()).collect();
    let idx = |names: &[&str]| cols.iter().position(|c| names.contains(&c.as_str()));
    let Some(pi) = idx(&["path", "file", "filename"]) else { return Ok(()) };
    let (ai, ci, li, si) = (idx(&["creator", "artist", "author"]), idx(&["copyright"]), idx(&["license", "licence"]), idx(&["source", "url"]));
    let tx = cat.conn.unchecked_transaction()?;
    for line in lines {
        let f = parse_csv_line(line);
        let get = |i: Option<usize>| i.and_then(|i| f.get(i)).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let Some(rel) = get(Some(pi)) else { continue };
        cat.conn.execute(
            "UPDATE file SET artist = COALESCE(?1, artist), copyright = COALESCE(?2, copyright), license = COALESCE(?3, license),
             source = COALESCE(?4, source) WHERE root_id = ?5 AND rel_path = ?6",
            params![get(ai), get(ci), get(li), get(si), root_id, rel],
        )?;
    }
    tx.commit()?;
    let _ = path_str(root);
    Ok(())
}

pub fn parse_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, q) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', _) => q = !q,
            (',', false) => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}
