//! Offline source-photo filters. Face boxes and colour grids come from the
//! index; document detection uses larger original images and screenshot names.
//! Optional model judgements are cached separately from the offline rules.

use crate::catalog::{Catalog, Hash};
use crate::color::dequantize_lab;
use crate::decode;
use crate::job::Job;
use crate::pool::{Pool, PoolPhoto};
use anyhow::{bail, Result};
use image::RgbImage;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Optional image judgement. Implementations must return an error on failed,
/// refused or malformed responses; those must never be cached as "photo".
pub trait DocumentClassifier {
    fn cache_key(&self) -> String;
    fn is_document(&self, image: &RgbImage, job: &Job) -> Result<bool>;
}

/// Optional semantic image tagger (vision LLM or local tagger).
/// Tags are cached permanently in SQLite per `(content_hash, cache_key)` so
/// photos are never re-tagged across sessions.
pub trait SemanticTagger {
    fn cache_key(&self) -> String;
    fn tag_image(&self, image: &RgbImage, job: &Job) -> Result<Vec<String>>;
}

pub const DOCUMENT_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum People {
    #[default]
    Any,
    Portraits,
    NoPeople,
}

impl People {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "any" | "all" => Ok(Self::Any),
            "portraits" | "people" => Ok(Self::Portraits),
            "no-people" | "no_people" | "none" => Ok(Self::NoPeople),
            _ => bail!("unknown people filter '{s}' (any, portraits, no-people)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Colour {
    #[default]
    Any,
    Colour,
    BlackAndWhite,
}

impl Colour {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "any" | "all" => Ok(Self::Any),
            "colour" | "color" => Ok(Self::Colour),
            "bw" | "b&w" | "black-and-white" | "black_and_white" | "monochrome" => Ok(Self::BlackAndWhite),
            _ => bail!("unknown colour filter '{s}' (any, colour, bw)"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoFilters {
    pub people: People,
    pub colour: Colour,
    pub skip_documents: bool,
    /// Earliest EXIF taken date to keep (`YYYY`, `YYYY-MM`, or `YYYY-MM-DD`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_from: Option<String>,
    /// Latest EXIF taken date to keep (`YYYY`, `YYYY-MM`, or `YYYY-MM-DD`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_to: Option<String>,
    /// Keep photos whose EXIF camera name contains this text (case-insensitive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    /// Keep photos whose EXIF lens name contains this text (case-insensitive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens: Option<String>,
    /// Keep photos matching a semantic theme query against cached AI tags and folder/file names
    /// (e.g. "beach sunset, ocean" or "dogs outdoors").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme_query: Option<String>,
}

fn non_empty(o: &Option<String>) -> Option<&str> {
    o.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

/// Parses `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` (also accepting `:` or `/`) into `YYYY-MM-DD`.
pub fn parse_date_bound(s: &str, end: bool) -> Result<String> {
    let norm = s.trim().replace([':', '/'], "-");
    let parts: Vec<&str> = norm.split('-').collect();
    let parse_u = |p: &str, min_len: usize, max_len: usize| -> Option<u32> {
        if p.len() < min_len || p.len() > max_len || !p.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        p.parse().ok()
    };
    let (y, m, d) = match parts.as_slice() {
        [y] => (parse_u(y, 4, 4), Some(if end { 12 } else { 1 }), Some(if end { 31 } else { 1 })),
        [y, m] => (parse_u(y, 4, 4), parse_u(m, 1, 2), Some(if end { 31 } else { 1 })),
        [y, m, d] => (parse_u(y, 4, 4), parse_u(m, 1, 2), parse_u(d, 1, 2)),
        _ => (None, None, None),
    };
    match (y, m, d) {
        (Some(y), Some(m @ 1..=12), Some(d @ 1..=31)) if y > 0 => Ok(format!("{y:04}-{m:02}-{d:02}")),
        _ => bail!("date '{}' must be YYYY, YYYY-MM or YYYY-MM-DD", s.trim()),
    }
}

fn exif_date_ymd(taken: &str) -> Option<String> {
    let t = taken.trim();
    let num = |a: usize, b: usize| t.get(a..b).and_then(|s| s.parse::<u32>().ok());
    match (num(0, 4), num(5, 7), num(8, 10)) {
        (Some(y), Some(m @ 1..=12), Some(d @ 1..=31)) if y > 0 => Some(format!("{y:04}-{m:02}-{d:02}")),
        _ => None,
    }
}

impl PhotoFilters {
    pub fn has_exif_filter(&self) -> bool {
        non_empty(&self.date_from).is_some()
            || non_empty(&self.date_to).is_some()
            || non_empty(&self.camera).is_some()
            || non_empty(&self.lens).is_some()
    }

    pub fn has_theme_filter(&self) -> bool {
        non_empty(&self.theme_query).is_some()
    }

    pub fn is_default(&self) -> bool {
        self.people == People::Any
            && self.colour == Colour::Any
            && !self.skip_documents
            && !self.has_exif_filter()
            && !self.has_theme_filter()
    }

    pub fn validate(&self) -> Result<()> {
        let from = non_empty(&self.date_from).map(|s| parse_date_bound(s, false)).transpose()?;
        let to = non_empty(&self.date_to).map(|s| parse_date_bound(s, true)).transpose()?;
        if let (Some(a), Some(b)) = (&from, &to) {
            if a > b {
                bail!("start date ({}) must not be after end date ({})", self.date_from.as_deref().unwrap_or(""), self.date_to.as_deref().unwrap_or(""));
            }
        }
        Ok(())
    }

    pub fn apply(&self, pool: &Pool, cat: &Catalog, libs: &[i64], merge: bool, job: &Job) -> Result<Pool> {
        self.apply_with_classifier(pool, cat, libs, merge, None, job)
    }

    pub fn apply_with_classifier(&self, pool: &Pool, cat: &Catalog, libs: &[i64], merge: bool, classifier: Option<&dyn DocumentClassifier>, job: &Job) -> Result<Pool> {
        self.validate()?;
        let exif_allowed = if self.has_exif_filter() { Some(exif_matching_hashes(cat, libs, self)?) } else { None };
        let theme_q = non_empty(&self.theme_query);
        let (photo_tags, path_hints) = if theme_q.is_some() {
            (Some(cat.all_photo_tags(libs)?), Some(photo_path_hints(cat, libs)?))
        } else {
            (None, None)
        };
        let screenshots = if self.skip_documents { screenshot_hashes(cat, libs)? } else { HashSet::new() };
        let offline_key = format!("offline:{DOCUMENT_VERSION}");
        let cache_key = if self.skip_documents {
            match classifier {
                Some(c) => {
                    let key = c.cache_key();
                    cat.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('document_classifier', ?1)", [&key])?;
                    key
                }
                None => cat.conn.query_row("SELECT value FROM meta WHERE key = 'document_classifier'", [], |r| r.get::<_, String>(0)).optional()?.unwrap_or_else(|| offline_key.clone()),
            }
        } else { offline_key.clone() };
        let mut photos = Vec::new();
        let mut unreadable = 0usize;
        for (i, p) in pool.photos.iter().enumerate() {
            if classifier.is_some() || i % 128 == 0 {
                job.check()?;
                job.report("filtering source photos (reusing saved checks)", i as u64, pool.photos.len() as u64);
            }
            if !self.allows(p) { continue; }
            if let Some(allowed) = &exif_allowed {
                if !allowed.contains(&p.hash) { continue; }
            }
            if let Some(q) = theme_q {
                let tags = photo_tags.as_ref().and_then(|m| m.get(&p.hash)).map(|v| v.as_slice()).unwrap_or(&[]);
                let hint = path_hints.as_ref().and_then(|m| m.get(&p.hash)).map(|s| s.as_str());
                if !matches_theme(q, tags, hint) {
                    continue;
                }
            }
            if self.skip_documents {
                if screenshots.contains(&p.hash) { continue; }
                let cached = cached_document(cat, &p.hash, &cache_key)?.map(|v| (v, cache_key != offline_key));
                let cached = match cached {
                    None if classifier.is_none() && cache_key != offline_key => cached_document(cat, &p.hash, &offline_key)?.map(|v| (v, false)),
                    v => v,
                };
                let (document, from_model) = match cached {
                    Some(v) => v,
                    None => {
                        let Ok(image) = analysis_image(cat, p, libs) else {
                            unreadable += 1;
                            continue;
                        };
                        job.check()?;
                        let document = match classifier {
                            Some(c) => {
                                job.report("checking uncached photos with vision model", i as u64, pool.photos.len() as u64);
                                c.is_document(&image, job)?
                            }
                            None => likely_document(&image),
                        };
                        let decision_key = if classifier.is_some() { &cache_key } else { &offline_key };
                        cat.conn.execute("INSERT OR REPLACE INTO document_decision(content_hash, classifier, document) VALUES(?1, ?2, ?3)", rusqlite::params![&p.hash[..], decision_key, document])?;
                        job.check()?;
                        (document, classifier.is_some())
                    }
                };
                // A saved AI decision keeps its meaning without a connection.
                // Only offline rules allow a face to override a document result.
                if document && (from_model || !p.faces.iter().any(|f| f.rect[2] * f.rect[3] >= 0.02)) { continue; }
            }
            photos.push(p.clone());
        }
        if photos.is_empty() {
            if unreadable > 0 {
                bail!("cannot inspect unchanged original photos for the document filter; reconnect or rescan your folders, or turn off Skip likely documents/screenshots");
            }
            bail!("no photos match these filters; relax the people, colour, date/camera, theme or document filters, or add more photos");
        }
        job.check()?;
        let mut out = Pool { photos, group_count: 0 };
        out.assign_groups(merge);
        Ok(out)
    }

    fn allows(&self, p: &PoolPhoto) -> bool {
        let people = match self.people {
            People::Any => true,
            People::Portraits => p.faces.iter().any(|f| f.rect[2] * f.rect[3] >= 0.02),
            People::NoPeople => p.faces.is_empty(),
        };
        let colour = match self.colour {
            Colour::Any => true,
            Colour::Colour => !is_monochrome(p),
            Colour::BlackAndWhite => is_monochrome(p),
        };
        people && colour
    }
}

fn exif_matching_hashes(cat: &Catalog, libs: &[i64], f: &PhotoFilters) -> Result<HashSet<Hash>> {
    if libs.is_empty() {
        return Ok(HashSet::new());
    }
    let from = non_empty(&f.date_from).map(|s| parse_date_bound(s, false)).transpose()?;
    let to = non_empty(&f.date_to).map(|s| parse_date_bound(s, true)).transpose()?;
    let cam = non_empty(&f.camera).map(|s| s.to_lowercase());
    let lens = non_empty(&f.lens).map(|s| s.to_lowercase());
    let ids = libs.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    let mut stmt = cat.conn.prepare(&format!(
        "SELECT f.content_hash, f.taken, f.camera, f.lens FROM file f JOIN root r ON r.id = f.root_id WHERE f.status = 0 AND r.library_id IN ({ids})"
    ))?;
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, Option<String>>(3)?))
    })?;
    let mut out = HashSet::new();
    for row in rows {
        let (hash, taken, camera, photo_lens) = row?;
        if from.is_some() || to.is_some() {
            let Some(ymd) = taken.as_deref().and_then(exif_date_ymd) else { continue };
            if let Some(a) = &from {
                if ymd < *a { continue; }
            }
            if let Some(b) = &to {
                if ymd > *b { continue; }
            }
        }
        if let Some(needle) = &cam {
            if !camera.as_deref().is_some_and(|c| c.to_lowercase().contains(needle)) {
                continue;
            }
        }
        if let Some(needle) = &lens {
            if !photo_lens.as_deref().is_some_and(|l| l.to_lowercase().contains(needle)) {
                continue;
            }
        }
        if let Ok(h) = hash.try_into() {
            out.insert(h);
        }
    }
    Ok(out)
}

fn cached_document(cat: &Catalog, hash: &Hash, key: &str) -> Result<Option<bool>> {
    Ok(cat.conn.query_row("SELECT document FROM document_decision WHERE content_hash = ?1 AND classifier = ?2", rusqlite::params![&hash[..], key], |r| r.get(0)).optional()?)
}

/// Read the upright original, then bound its size. The 96px preview thumbnail
/// loses handwriting and small browser text, so it cannot classify documents.
pub fn analysis_image(cat: &Catalog, p: &PoolPhoto, libs: &[i64]) -> Result<RgbImage> {
    for location in cat.locations(&p.hash, libs)? {
        let Ok(meta) = std::fs::metadata(&location.path) else { continue };
        if meta.len() > decode::MAX_FILE_BYTES { continue; }
        let Ok(bytes) = std::fs::read(&location.path) else { continue };
        if blake3::hash(&bytes).as_bytes() != &p.hash { continue; }
        if let Ok(decoded) = decode::decode_bytes(&bytes, Some(512)) {
            return Ok(bounded_image(&decoded.image));
        }
    }
    bail!("cannot inspect an unchanged original photo for the document filter; reconnect or rescan its folder, or turn off Skip likely documents/screenshots")
}

pub fn bounded_image(image: &RgbImage) -> RgbImage {
    let scale = (1024.0 / image.width().max(image.height()).max(1) as f64).min(1.0);
    let (w, h) = (((image.width() as f64 * scale).round() as u32).max(1), ((image.height() as f64 * scale).round() as u32).max(1));
    image::imageops::thumbnail(image, w, h)
}

fn is_monochrome(p: &PoolPhoto) -> bool {
    // Cell chroma tolerates quantisation and slight scanner/JPEG colour casts.
    let neutral = p.grid.chunks(3).filter(|c| {
        let lab = dequantize_lab([c[0], c[1], c[2]]);
        lab[1] * lab[1] + lab[2] * lab[2] <= 25.0
    }).count();
    neutral * 100 >= p.grid.len() / 3 * 95
}

fn screenshot_hashes(cat: &Catalog, libs: &[i64]) -> Result<HashSet<Hash>> {
    let ids = libs.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    let mut stmt = cat.conn.prepare(&format!("SELECT f.content_hash, f.rel_path FROM file f JOIN root r ON r.id = f.root_id WHERE f.status = 0 AND r.library_id IN ({ids})"))?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, String>(1)?)))?;
    let mut out = HashSet::new();
    for row in rows {
        let (hash, path) = row?;
        let name = std::path::Path::new(&path).file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        let screenshot = ["screenshot", "screen shot", "screen_shot", "screen-shot", "skærmbillede", "skjermbilde", "skärmbild", "bildschirmfoto", "capture d’écran", "capture d'ecran"].iter().any(|s| name.contains(s));
        if !screenshot { continue; }
        let Ok(hash) = hash.try_into() else { continue };
        out.insert(hash);
    }
    Ok(out)
}

/// Heuristic for pages and flat UI captures, deliberately not an OCR filter:
/// broad flat backgrounds plus many short, high-contrast text-like runs.
fn likely_document(img: &RgbImage) -> bool {
    let small = bounded_image(img);
    let (w, h) = small.dimensions();
    if w < 24 || h < 24 { return false; }
    let distance = |a: image::Rgb<u8>, b: image::Rgb<u8>| (0..3).map(|k| a[k].abs_diff(b[k]) as u32).sum::<u32>();
    let mut flat = 0u32;
    let mut text_rows = 0u32;
    let mut edges = 0u32;
    let mut paper = 0u32;
    for y in 1..h {
        let mut transitions = 0;
        for x in 1..w {
            let p = *small.get_pixel(x, y);
            let left = *small.get_pixel(x - 1, y);
            let above = *small.get_pixel(x, y - 1);
            let lo = *p.0.iter().min().unwrap();
            let hi = *p.0.iter().max().unwrap();
            if lo > 150 && hi - lo < 25 { paper += 1; }
            if distance(p, left) < 18 && distance(p, above) < 18 { flat += 1; }
            if distance(p, left) > 150 { transitions += 1; edges += 1; }
        }
        if transitions >= 8 && transitions < w / 2 { text_rows += 1; }
    }
    let n = ((w - 1) * (h - 1)) as f32;
    if flat as f32 / n > 0.62 && text_rows as f32 / h as f32 > 0.12 && edges as f32 / n > 0.015 { return true; }
    if paper as f32 / n <= 0.85 { return false; }
    // Compare against the surrounding paper rather than an absolute dark-ink
    // threshold. This catches pale handwriting, including sideways notes.
    let background = image::imageops::blur(&small, 6.0);
    let shade = |c: &image::Rgb<u8>| (c[0] as f32 + c[1] as f32 * 2.0 + c[2] as f32) / 4.0;
    let dark_ink = small.pixels().zip(background.pixels()).filter(|(p, bg)| shade(bg) - shade(p) > 8.0).count() as f32;
    let density = dark_ink / (w * h) as f32;
    density > 0.002 && density < 0.12
}

/// Tags every photo in `pool` not yet cached under `tagger.cache_key()`,
/// persisting results in SQLite `photo_tags`. Returns the count newly tagged.
pub fn tag_untagged_pool(cat: &Catalog, pool: &Pool, libs: &[i64], tagger: &dyn SemanticTagger, job: &Job) -> Result<usize> {
    let key = tagger.cache_key();
    let total = pool.photos.len() as u64;
    let mut newly_tagged = 0usize;
    for (i, p) in pool.photos.iter().enumerate() {
        job.check()?;
        if cat.photo_tags(&p.hash, Some(&key))?.is_some() {
            if i % 64 == 0 {
                job.report("tagging photos with AI (reusing cached tags)", i as u64, total);
            }
            continue;
        }
        job.report("tagging photos with AI", i as u64, total);
        let img = match analysis_image(cat, p, libs) {
            Ok(img) => img,
            Err(_) => {
                let Some(thumb) = cat.thumb(&p.hash)? else { continue };
                let Ok(dec) = decode::decode_bytes(&thumb, None) else { continue };
                dec.image
            }
        };
        job.check()?;
        let tags = tagger.tag_image(&img, job)?;
        cat.put_photo_tags(&p.hash, &key, &tags)?;
        newly_tagged += 1;
    }
    job.report("tagging photos with AI", total, total);
    Ok(newly_tagged)
}

fn photo_path_hints(cat: &Catalog, libs: &[i64]) -> Result<HashMap<Hash, String>> {
    if libs.is_empty() {
        return Ok(HashMap::new());
    }
    let ids = libs.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    let mut stmt = cat.conn.prepare(&format!(
        "SELECT f.content_hash, r.path, f.rel_path FROM file f JOIN root r ON r.id = f.root_id WHERE f.status = 0 AND r.library_id IN ({ids})"
    ))?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?;
    let mut out: HashMap<Hash, String> = HashMap::new();
    for row in rows {
        let (hb, root, rel) = row?;
        let Ok(h) = hb.try_into() else { continue };
        let root_name = std::path::Path::new(&root).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let entry = out.entry(h).or_default();
        if !entry.is_empty() {
            entry.push(' ');
        }
        entry.push_str(&root_name);
        entry.push(' ');
        entry.push_str(&rel);
    }
    Ok(out)
}

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "by", "from", "photos", "photo", "pictures", "picture", "only", "my",
];

const SYNONYM_GROUPS: &[&[&str]] = &[
    &["beach", "ocean", "sea", "coast", "shore", "sand", "wave", "waves", "surf", "seaside"],
    &["sunset", "sunrise", "golden", "dusk", "dawn", "twilight", "evening"],
    &["mountain", "mountains", "hill", "hills", "peak", "alpine", "hike", "hiking", "trail"],
    &["forest", "woods", "tree", "trees", "woodland", "jungle", "nature", "foliage"],
    &["snow", "winter", "ice", "ski", "skiing", "frost", "snowy"],
    &["dog", "dogs", "puppy", "puppies", "canine", "pet", "pets"],
    &["cat", "cats", "kitten", "kittens", "feline", "pet", "pets"],
    &["bird", "birds", "wildlife", "animal", "animals"],
    &["flower", "flowers", "blossom", "bloom", "garden", "rose", "tulip", "floral"],
    &["city", "urban", "street", "building", "buildings", "skyline", "architecture", "downtown"],
    &["food", "meal", "dinner", "lunch", "breakfast", "restaurant", "cafe", "coffee", "cake", "dessert"],
    &["car", "cars", "vehicle", "road", "drive", "motorcycle", "bike", "bicycle"],
    &["night", "stars", "moon", "fireworks", "neon", "dark"],
    &["family", "kids", "child", "children", "baby", "wedding", "party", "celebration", "portrait", "people"],
    &["water", "lake", "river", "waterfall", "pond", "pool", "boat", "sailing"],
    &["autumn", "fall", "leaves", "harvest"],
    &["summer", "vacation", "holiday", "travel", "trip", "sunny"],
];

fn tokenize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn stem_matches(token: &str, needle: &str) -> bool {
    if token == needle || token.contains(needle) {
        return true;
    }
    if needle.len() >= 4 {
        let stripped = needle.strip_suffix('s').unwrap_or(needle);
        if token == stripped || token.starts_with(stripped) {
            return true;
        }
    }
    false
}

fn term_matches_haystack(term: &str, words: &[String], full: &str) -> bool {
    if words.iter().any(|w| stem_matches(w, term)) || full.contains(term) {
        return true;
    }
    for group in SYNONYM_GROUPS {
        if group.iter().any(|&g| g == term || stem_matches(g, term)) {
            if group.iter().any(|&syn| words.iter().any(|w| stem_matches(w, syn))) {
                return true;
            }
        }
    }
    false
}

/// Matches a natural-language theme query against a photo's cached AI tags and
/// optional folder/filename hint.
///
/// Comma-separated or ` or `-separated clauses act as OR alternatives; words
/// within a clause act as AND terms (with photographic synonym expansion).
pub fn matches_theme(query: &str, tags: &[String], path_hint: Option<&str>) -> bool {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return true;
    }
    let mut full = tags.join(" ").to_lowercase();
    if let Some(p) = path_hint {
        full.push(' ');
        full.push_str(&p.to_lowercase());
    }
    let words = tokenize_words(&full);
    if words.is_empty() {
        return false;
    }
    let normalized = q.replace(" or ", ",");
    for clause in normalized.split(',') {
        let terms: Vec<String> = tokenize_words(clause)
            .into_iter()
            .filter(|w| !STOP_WORDS.contains(&w.as_str()))
            .collect();
        if terms.is_empty() {
            continue;
        }
        if terms.iter().all(|t| term_matches_haystack(t, &words, &full)) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{catalog::PoolRow, descriptor, faces::Face};

    /// Private examples stay outside the repository. Supply paths separated by
    /// the platform path separator with MOSAIC_DOCUMENT_SAMPLES.
    #[test]
    #[ignore]
    fn document_samples_are_excluded() {
        let paths = std::env::var_os("MOSAIC_DOCUMENT_SAMPLES").expect("set MOSAIC_DOCUMENT_SAMPLES");
        let tmp = tempfile::tempdir().unwrap();
        let photos = tmp.path().join("photos");
        std::fs::create_dir(&photos).unwrap();
        let mut hashes = Vec::new();
        for (i, path) in std::env::split_paths(&paths).enumerate() {
            let bytes = std::fs::read(path).unwrap();
            hashes.push(*blake3::hash(&bytes).as_bytes());
            std::fs::write(photos.join(format!("sample-{i}.png")), bytes).unwrap();
        }
        RgbImage::from_fn(800, 600, |x, y| image::Rgb([(80 + x / 8) as u8, (60 + y / 8) as u8, 70])).save(photos.join("ordinary-photo.png")).unwrap();
        let cat = Catalog::open(&tmp.path().join("data")).unwrap();
        let lib = cat.create_library("Private regression samples").unwrap();
        cat.add_root(lib, &photos).unwrap();
        crate::indexer::update_library(&cat, lib, &crate::indexer::IndexOptions { threads: 2, ..Default::default() }, &Job::default()).unwrap();
        let pool = Pool::load(&cat, &[lib], true).unwrap();
        let filters = PhotoFilters { skip_documents: true, ..Default::default() };
        let kept = filters.apply(&pool, &cat, &[lib], true, &Job::default()).unwrap();
        let missed: Vec<_> = hashes.iter().enumerate().filter(|(_, h)| kept.index_of(h).is_some()).map(|(i, _)| i + 1).collect();
        assert!(missed.is_empty(), "documents/screenshots still admitted: sample numbers {missed:?}");
        assert_eq!(kept.photos.len(), 1, "the ordinary photo must remain available");
    }

    #[test]
    fn people_and_colour_filters_combine() {
        let gray = RgbImage::from_pixel(100, 100, image::Rgb([110, 110, 110]));
        let colour = RgbImage::from_pixel(100, 100, image::Rgb([200, 40, 20]));
        let p = Pool::from_rows(vec![PoolRow { hash: [1; 32], width: 100, height: 100, grid: descriptor::compute(&gray, 100, 100).grid, dhash: 0, faces: vec![Face { rect: [0.3, 0.2, 0.3, 0.3], score: 1.0 }], saliency: None }], false);
        let mut f = PhotoFilters { people: People::Portraits, colour: Colour::BlackAndWhite, skip_documents: false, ..Default::default() };
        assert!(f.allows(&p.photos[0]));
        f.colour = Colour::Colour;
        assert!(!f.allows(&p.photos[0]));
        let mut photo = p.photos[0].clone();
        photo.grid = descriptor::compute(&colour, 100, 100).grid;
        assert!(f.allows(&photo));
        f.people = People::NoPeople;
        assert!(!f.allows(&photo));
        photo.faces.clear();
        assert!(f.allows(&photo));
    }

    #[test]
    fn date_bounds_parse_and_validate() {
        assert_eq!(parse_date_bound("2023", false).unwrap(), "2023-01-01");
        assert_eq!(parse_date_bound("2023", true).unwrap(), "2023-12-31");
        assert_eq!(parse_date_bound("2023-7", false).unwrap(), "2023-07-01");
        assert_eq!(parse_date_bound("2023/07/14", false).unwrap(), "2023-07-14");
        assert!(parse_date_bound("2023-13-01", false).is_err());
        assert!(parse_date_bound("yesterday", false).is_err());
        assert_eq!(exif_date_ymd("2023:07:14 15:02:11").as_deref(), Some("2023-07-14"));
        assert_eq!(exif_date_ymd("0000:00:00 00:00:00"), None);
        let bad = PhotoFilters { date_from: Some("2025".into()), date_to: Some("2024".into()), ..Default::default() };
        assert!(bad.validate().is_err());
        let ok = PhotoFilters { date_from: Some("2022".into()), date_to: Some("2024-06".into()), ..Default::default() };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn document_heuristic_keeps_flat_photos_and_detects_a_page() {
        let mut page = RgbImage::from_pixel(192, 192, image::Rgb([255, 255, 255]));
        assert!(!likely_document(&page));
        for line in 0..16 {
            for word in 0..8 {
                for y in 20 + line * 9..23 + line * 9 {
                    for x in 12 + word * 20..24 + word * 20 { page.put_pixel(x, y, image::Rgb([25, 25, 25])); }
                }
            }
        }
        assert!(likely_document(&page));
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 85).encode(page.as_raw(), 192, 192, jpeg_encoder::ColorType::Rgb).unwrap();
        assert!(likely_document(&decode::decode_bytes(&jpeg, None).unwrap().image), "JPEG compression must not hide a page at analysis resolution");
        assert!(!likely_document(&RgbImage::from_fn(192, 192, |x, y| image::Rgb([x as u8, y as u8, 100]))));
    }

    #[test]
    fn faint_sparse_notes_are_detected_in_both_orientations() {
        let mut page = RgbImage::from_pixel(512, 512, image::Rgb([225, 228, 233]));
        for line in [100, 230, 360] {
            for x in 65..440 {
                let y = line + ((x as f32 / 8.0).sin() * 10.0) as i32;
                for d in 0..3 { page.put_pixel(x, (y + d) as u32, image::Rgb([155, 177, 199])); }
            }
        }
        assert!(likely_document(&page));
        assert!(likely_document(&image::imageops::rotate90(&page)));
        assert!(!likely_document(&RgbImage::from_pixel(512, 512, image::Rgb([225, 228, 233]))));
        let bounded = bounded_image(&RgbImage::new(2000, 1000));
        assert_eq!(bounded.dimensions(), (1024, 512), "model previews must keep the photo's aspect ratio");
    }

    #[test]
    fn theme_query_matches_tags_synonyms_and_paths() {
        let beach_tags = vec!["ocean".into(), "golden hour".into(), "waves".into()];
        assert!(matches_theme("beach sunset", &beach_tags, None));
        assert!(matches_theme("mountain, beach", &beach_tags, None));
        assert!(matches_theme("snow or coast", &beach_tags, None));
        assert!(!matches_theme("snow mountains", &beach_tags, None));
        // Works offline via folder/filename hints even before AI tagging.
        assert!(matches_theme("dogs outdoors", &[], Some("2024_puppy_park/outdoors_01.jpg")));
        assert!(!matches_theme("cat indoors", &[], Some("2024_puppy_park/outdoors_01.jpg")));
    }
}
