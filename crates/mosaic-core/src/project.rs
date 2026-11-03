//! Mosaic projects: everything needed to reproduce and re-render a mosaic.
//! Projects are JSON files stored wherever the user chooses, separate from
//! the catalogue (cached analysis) and from the source photos.

use crate::atomic::TempOutput;
use crate::catalog::{hex, unhex, Catalog, Hash};
use crate::color::{dequantize_lab, quantize_lab, Lab};
use crate::decode;
use crate::descriptor::{CELLS, DESCRIPTOR_VERSION};
use crate::job::Job;
use crate::matching::{GridSpec, MatchError, MatchSettings, MatchStats, Target, TileAssign};
use crate::matching::match_tiles_masked;
use crate::pool::Pool;
use crate::shape::Outline;
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const PROJECT_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifRef {
    pub path: PathBuf,
    pub content_hash: String,
    pub width: u32,
    pub height: u32,
    /// Normalised part of the motif that the mosaic covers (cover fit, centred).
    pub crop: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryRef {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub hash: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub format: u32,
    pub app_version: String,
    pub motif: MotifRef,
    pub libraries: Vec<LibraryRef>,
    pub grid: GridSpec,
    pub settings: MatchSettings,
    pub descriptor_version: u32,
    pub sources: Vec<SourceRef>,
    /// Row-major tile assignments; `photo` indexes `sources`.
    pub tiles: Vec<TileAssign>,
    /// Target motif cells (CELLS×CELLS per tile), quantised Lab, base64.
    pub target_cells: String,
    pub stats: MatchStats,
    /// Mosaic outline; tiles outside it are [`TileAssign::EMPTY`]. None = full rectangle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline: Option<Outline>,
}

/// Load the motif and sample it for the grid.
pub fn load_target(motif: &Path, grid: &GridSpec) -> Result<(Target, MotifRef)> {
    let bytes = std::fs::read(motif).with_context(|| format!("cannot read motif {}", motif.display()))?;
    let need = (grid.cols.max(grid.rows) as usize * CELLS * 2) as u32;
    let d = decode::decode_bytes(&bytes, Some(need)).with_context(|| format!("cannot decode motif {}", motif.display()))?;
    let (target, crop) = Target::from_image(&d.image, grid);
    let mref = MotifRef {
        path: std::fs::canonicalize(motif).unwrap_or_else(|_| motif.to_path_buf()),
        content_hash: hex(blake3::hash(&bytes).as_bytes()),
        width: d.info.full_width,
        height: d.info.full_height,
        crop,
    };
    Ok((target, mref))
}

impl Project {
    /// Match and build a project from an already loaded pool.
    pub fn create(
        pool: &Pool,
        libraries: Vec<LibraryRef>,
        target: &Target,
        motif: MotifRef,
        grid: GridSpec,
        settings: MatchSettings,
        job: &Job,
    ) -> Result<Project, MatchError> {
        Self::create_with_outline(pool, libraries, target, motif, grid, settings, None, job)
    }

    /// Like [`Project::create`], limited to the tiles inside `outline`.
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_outline(
        pool: &Pool,
        libraries: Vec<LibraryRef>,
        target: &Target,
        motif: MotifRef,
        grid: GridSpec,
        settings: MatchSettings,
        outline: Option<Outline>,
        job: &Job,
    ) -> Result<Project, MatchError> {
        let active = outline.as_ref().map(|o| o.active_tiles(&grid)).transpose()?;
        let (tiles, stats) = match_tiles_masked(pool, target, &grid, &settings, active.as_deref(), job)?;
        // Keep only the photos actually used, in first-use order.
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut sources = Vec::new();
        let tiles = tiles
            .into_iter()
            .map(|mut t| {
                if t.is_empty() {
                    return t;
                }
                let next = remap.len() as u32;
                let idx = *remap.entry(t.photo).or_insert_with(|| {
                    let p = &pool.photos[t.photo as usize];
                    sources.push(SourceRef { hash: hex(&p.hash), width: p.width, height: p.height });
                    next
                });
                t.photo = idx;
                t
            })
            .collect();
        Ok(Project {
            format: PROJECT_FORMAT,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            motif,
            libraries,
            grid,
            settings,
            descriptor_version: DESCRIPTOR_VERSION,
            sources,
            tiles,
            target_cells: encode_cells(&target.cells()),
            stats,
            outline,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out = TempOutput::create(path)?;
        serde_json::to_writer(std::io::BufWriter::new(out.file()), self)?;
        out.file().flush()?;
        out.commit()
    }

    pub fn load(path: &Path) -> Result<Project> {
        let f = std::fs::File::open(path).with_context(|| format!("cannot open project {}", path.display()))?;
        let p: Project = serde_json::from_reader(std::io::BufReader::new(f)).with_context(|| format!("invalid project file {}", path.display()))?;
        if p.format != PROJECT_FORMAT {
            bail!("unsupported project format {}", p.format);
        }
        if p.tiles.len() != p.grid.tiles() {
            bail!("project is inconsistent: {} tiles for a {}×{} grid", p.tiles.len(), p.grid.cols, p.grid.rows);
        }
        if p.tiles.iter().any(|t| if t.is_empty() { p.outline.is_none() } else { t.photo as usize >= p.sources.len() }) {
            bail!("project is inconsistent: tile refers to unknown source");
        }
        Ok(p)
    }

    /// Number of tiles showing a photo (all tiles unless an outline is set).
    pub fn placed_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| !t.is_empty()).count()
    }

    /// Whether any photo tile overlaps the artwork window [x0, x1) × [y0, y1) in mm.
    pub fn window_has_tiles(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> bool {
        if self.outline.is_none() {
            return true;
        }
        let g = &self.grid;
        let (tw, th) = g.tile_mm();
        let c0 = ((x0 / tw).floor().max(0.0) as u32).min(g.cols);
        let c1 = ((x1 / tw).ceil().max(0.0) as u32).min(g.cols);
        let r0 = ((y0 / th).floor().max(0.0) as u32).min(g.rows);
        let r1 = ((y1 / th).ceil().max(0.0) as u32).min(g.rows);
        (r0..r1).any(|r| (c0..c1).any(|c| !self.tiles[(r * g.cols + c) as usize].is_empty()))
    }

    pub fn target_cells(&self) -> Result<Vec<Lab>> {
        decode_cells(&self.target_cells, self.grid.cols as usize * CELLS * self.grid.rows as usize * CELLS)
    }

    pub fn source_hashes(&self) -> Result<Vec<Hash>> {
        self.sources.iter().map(|s| unhex(&s.hash)).collect()
    }

    pub fn library_ids(&self) -> Vec<i64> {
        self.libraries.iter().map(|l| l.id).collect()
    }

    /// Uses per source index.
    pub fn usage(&self) -> Vec<u32> {
        let mut u = vec![0u32; self.sources.len()];
        for t in self.tiles.iter().filter(|t| !t.is_empty()) {
            u[t.photo as usize] += 1;
        }
        u
    }
}

fn encode_cells(cells: &[Lab]) -> String {
    let mut b = Vec::with_capacity(cells.len() * 3);
    for c in cells {
        b.extend_from_slice(&quantize_lab(*c));
    }
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn decode_cells(s: &str, n: usize) -> Result<Vec<Lab>> {
    let b = base64::engine::general_purpose::STANDARD.decode(s)?;
    if b.len() != n * 3 {
        bail!("target cells have wrong size");
    }
    Ok(b.chunks_exact(3).map(|c| dequantize_lab([c[0], c[1], c[2]])).collect())
}

// ---------------------------------------------------------------------------
// Source validation before export.

#[derive(Debug, Clone, Serialize)]
pub enum ProblemKind {
    /// No file with this content exists in the project's libraries.
    NotIndexed,
    /// Indexed file(s) are missing on disk.
    Missing,
    /// Indexed file(s) exist but their content changed.
    Changed,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceProblem {
    pub source: usize,
    pub hash: String,
    pub kind: ProblemKind,
    pub paths: Vec<PathBuf>,
    pub tiles: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SourceCheck {
    pub resolved: Vec<Option<PathBuf>>,
    pub problems: Vec<SourceProblem>,
    pub rehashed: usize,
}

impl SourceCheck {
    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
    pub fn report(&self, p: &Project) -> String {
        if self.ok() {
            return format!("All {} source photos are present and unchanged.", p.sources.len());
        }
        let tiles: u32 = self.problems.iter().map(|x| x.tiles).sum();
        let mut s = format!("{} of {} source photos are unavailable ({} tiles affected):\n", self.problems.len(), p.sources.len(), tiles);
        for pr in self.problems.iter().take(20) {
            let where_ = pr.paths.first().map(|p| p.display().to_string()).unwrap_or_else(|| "(no indexed file)".into());
            s += &format!("  {:?}: {} [{} tiles] {}\n", pr.kind, &pr.hash[..12], pr.tiles, where_);
        }
        if self.problems.len() > 20 {
            s += &format!("  … and {} more\n", self.problems.len() - 20);
        }
        s += "Fix: reconnect or relink the source folder (library relink), restore the files, or update the library index.\n";
        s += "Re-matching the project would choose different photos; it is never done automatically.";
        s
    }
}

/// Resolve each source to an existing, unchanged file. Never substitutes.
pub fn check_sources(cat: &Catalog, p: &Project, job: &Job) -> Result<SourceCheck> {
    let libs = p.library_ids();
    let usage = p.usage();
    let mut out = SourceCheck::default();
    for (i, s) in p.sources.iter().enumerate() {
        if i % 256 == 0 {
            job.check()?;
            job.report("checking source photos", i as u64, p.sources.len() as u64);
        }
        let h = unhex(&s.hash)?;
        let locs = cat.locations(&h, &libs)?;
        let mut found = None;
        let mut kind = if locs.is_empty() { ProblemKind::NotIndexed } else { ProblemKind::Missing };
        for l in &locs {
            let Ok(m) = std::fs::metadata(&l.path) else { continue };
            let mtime = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as i64).unwrap_or(0);
            if m.len() as i64 == l.size && mtime == l.mtime_ns {
                found = Some(l.path.clone());
                break;
            }
            // Stat differs: verify content.
            out.rehashed += 1;
            match std::fs::read(&l.path) {
                Ok(b) if blake3::hash(&b).as_bytes() == &h => {
                    found = Some(l.path.clone());
                    break;
                }
                Ok(_) => kind = ProblemKind::Changed,
                Err(_) => {}
            }
        }
        if found.is_none() {
            out.problems.push(SourceProblem { source: i, hash: s.hash.clone(), kind, paths: locs.iter().map(|l| l.path.clone()).collect(), tiles: usage[i] });
        }
        out.resolved.push(found);
    }
    Ok(out)
}

/// CSV manifest of photos actually used, with supplied attribution metadata.
pub fn write_manifest(cat: &Catalog, p: &Project, out: &Path) -> Result<usize> {
    use crate::indexer::csv_field;
    let libs = p.library_ids();
    let usage = p.usage();
    let mut t = TempOutput::create(out)?;
    {
        let mut w = std::io::BufWriter::new(t.file());
        writeln!(w, "content_hash,uses,path,library,artist,copyright,license,source,width,height,camera,lens,f_number,exposure_s,iso,focal_mm,taken")?;
        for (i, s) in p.sources.iter().enumerate() {
            let h = unhex(&s.hash)?;
            let locs = cat.locations(&h, &libs)?;
            let l = locs.first();
            let a = l.map(|l| l.attribution.clone()).unwrap_or_default();
            let f = |o: &Option<String>| csv_field(o.as_deref().unwrap_or(""));
            let x = l.map(|l| l.exif.clone()).unwrap_or_default();
            let n = |o: Option<f64>| o.map(|v| format!("{v}")).unwrap_or_default();
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                s.hash,
                usage[i],
                csv_field(&l.map(|l| l.path.display().to_string()).unwrap_or_default()),
                csv_field(&l.map(|l| l.library.clone()).unwrap_or_default()),
                f(&a.artist),
                f(&a.copyright),
                f(&a.license),
                f(&a.source),
                s.width,
                s.height,
                f(&x.camera),
                f(&x.lens),
                n(x.f_number),
                n(x.exposure_s),
                x.iso.map(|v| v.to_string()).unwrap_or_default(),
                n(x.focal_mm),
                f(&x.taken)
            )?;
        }
        w.flush()?;
    }
    t.commit()?;
    Ok(p.sources.len())
}

pub fn libraries_refs(cat: &Catalog, ids: &[i64]) -> Result<Vec<LibraryRef>> {
    ids.iter().map(|&id| Ok(LibraryRef { id, name: cat.library_name(id).map_err(|_| anyhow!("library {id} not found"))? })).collect()
}

// ---------------------------------------------------------------------------
// Output statistics (written next to every export).

#[derive(Debug, Clone, Serialize)]
pub struct Range {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Counted {
    pub value: String,
    pub photos: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExifStats {
    /// Distinct photos with at least one EXIF camera field.
    pub photos_with_exif: usize,
    pub photos_without_exif: usize,
    pub aperture_f: Option<Range>,
    pub shutter_s: Option<Range>,
    pub iso: Option<Range>,
    pub focal_length_mm: Option<Range>,
    pub apertures: Vec<Counted>,
    pub shutter_speeds: Vec<Counted>,
    pub isos: Vec<Counted>,
    pub cameras: Vec<Counted>,
    pub lenses: Vec<Counted>,
    pub date_first: Option<String>,
    pub date_last: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputStats {
    pub output: String,
    pub output_kind: String,
    pub created_unix: u64,
    pub details: serde_json::Value,
    pub artwork_mm: (f64, f64),
    pub grid: (u32, u32),
    pub tiles: usize,
    pub distinct_photos_used: usize,
    pub max_uses_of_one_photo: u32,
    pub photos_used_once: usize,
    pub libraries: Vec<String>,
    pub method: String,
    pub seed: u64,
    pub max_reuse: Option<u32>,
    pub color_correction: f32,
    pub motif_blend: f32,
    pub mean_color_error_delta_e: f32,
    /// Weighted by tile count: how many tiles show photos taken with each setting.
    pub tiles_with_exif: u32,
    pub exif: ExifStats,
}

fn range(mut v: Vec<f64>) -> Option<Range> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(Range { min: v[0], median: v[v.len() / 2], max: v[v.len() - 1] })
}

fn top(mut m: HashMap<String, usize>, n: usize) -> Vec<Counted> {
    let mut v: Vec<Counted> = m.drain().map(|(value, photos)| Counted { value, photos }).collect();
    v.sort_by(|a, b| b.photos.cmp(&a.photos).then(a.value.cmp(&b.value)));
    v.truncate(n);
    v
}

pub fn shutter_label(s: f64) -> String {
    if s >= 1.0 { format!("{s:.1}s").replace(".0s", "s") } else { format!("1/{:.0}s", 1.0 / s) }
}

/// Statistics of the photos used in a project, including an EXIF summary.
pub fn output_stats(cat: &Catalog, p: &Project, output: &Path, kind: &str, details: serde_json::Value) -> Result<OutputStats> {
    let libs = p.library_ids();
    let usage = p.usage();
    let (mut ap, mut sh, mut iso, mut fl) = (vec![], vec![], vec![], vec![]);
    let (mut aps, mut shs, mut isos, mut cams, mut lenses) = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
    let (mut with, mut tiles_with, mut dates) = (0usize, 0u32, Vec::new());
    for (i, s) in p.sources.iter().enumerate() {
        let x = cat.locations(&unhex(&s.hash)?, &libs)?.into_iter().map(|l| l.exif).find(|x| !x.is_empty());
        let Some(x) = x else { continue };
        with += 1;
        tiles_with += usage[i];
        if let Some(v) = x.f_number {
            ap.push(v);
            *aps.entry(format!("f/{v:.1}")).or_insert(0) += 1;
        }
        if let Some(v) = x.exposure_s {
            sh.push(v);
            *shs.entry(shutter_label(v)).or_insert(0) += 1;
        }
        if let Some(v) = x.iso {
            iso.push(v as f64);
            *isos.entry(format!("ISO {v}")).or_insert(0) += 1;
        }
        if let Some(v) = x.focal_mm {
            fl.push(v);
        }
        if let Some(c) = x.camera {
            *cams.entry(c).or_insert(0) += 1;
        }
        if let Some(c) = x.lens {
            *lenses.entry(c).or_insert(0) += 1;
        }
        if let Some(d) = x.taken {
            dates.push(d);
        }
    }
    dates.sort();
    let exif = ExifStats {
        photos_with_exif: with,
        photos_without_exif: p.sources.len() - with,
        aperture_f: range(ap),
        shutter_s: range(sh),
        iso: range(iso),
        focal_length_mm: range(fl),
        apertures: top(aps, 10),
        shutter_speeds: top(shs, 10),
        isos: top(isos, 10),
        cameras: top(cams, 10),
        lenses: top(lenses, 10),
        date_first: dates.first().cloned(),
        date_last: dates.last().cloned(),
    };
    Ok(OutputStats {
        output: output.display().to_string(),
        output_kind: kind.to_string(),
        created_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        details,
        artwork_mm: (p.grid.width_mm, p.grid.height_mm),
        grid: (p.grid.cols, p.grid.rows),
        tiles: p.placed_tiles(),
        distinct_photos_used: p.sources.len(),
        max_uses_of_one_photo: usage.iter().copied().max().unwrap_or(0),
        photos_used_once: usage.iter().filter(|&&u| u == 1).count(),
        libraries: p.libraries.iter().map(|l| l.name.clone()).collect(),
        method: p.settings.method.name().into(),
        seed: p.settings.seed,
        max_reuse: p.settings.max_reuse,
        color_correction: p.settings.color_correction,
        motif_blend: p.settings.motif_blend,
        mean_color_error_delta_e: p.stats.mean_delta_e,
        tiles_with_exif: tiles_with,
        exif,
    })
}

/// Path of the statistics file written next to an export: `<output>.stats.json`.
pub fn stats_path(output: &Path) -> PathBuf {
    let mut s = output.as_os_str().to_owned();
    s.push(".stats.json");
    PathBuf::from(s)
}

pub fn write_output_stats(cat: &Catalog, p: &Project, output: &Path, kind: &str, details: serde_json::Value) -> Result<(PathBuf, OutputStats)> {
    let st = output_stats(cat, p, output, kind, details)?;
    let path = stats_path(output);
    let mut t = TempOutput::create(&path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(t.file()), &st)?;
    t.commit()?;
    Ok((path, st))
}
