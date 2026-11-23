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
use std::collections::HashSet;

/// Optional image judgement. Implementations must return an error on failed,
/// refused or malformed responses; those must never be cached as "photo".
pub trait DocumentClassifier {
    fn cache_key(&self) -> String;
    fn is_document(&self, image: &RgbImage, job: &Job) -> Result<bool>;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Colour {
    #[default]
    Any,
    Colour,
    BlackAndWhite,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoFilters {
    pub people: People,
    pub colour: Colour,
    pub skip_documents: bool,
}

impl PhotoFilters {
    pub fn apply(&self, pool: &Pool, cat: &Catalog, libs: &[i64], merge: bool, job: &Job) -> Result<Pool> {
        self.apply_with_classifier(pool, cat, libs, merge, None, job)
    }

    pub fn apply_with_classifier(&self, pool: &Pool, cat: &Catalog, libs: &[i64], merge: bool, classifier: Option<&dyn DocumentClassifier>, job: &Job) -> Result<Pool> {
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
        for (i, p) in pool.photos.iter().enumerate() {
            if classifier.is_some() || i % 128 == 0 {
                job.check()?;
                job.report("filtering source photos (reusing saved checks)", i as u64, pool.photos.len() as u64);
            }
            if !self.allows(p) { continue; }
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
                        let image = analysis_image(cat, p, libs)?;
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
            bail!("no photos match these filters; choose All photos / All colours, turn off Skip likely documents/screenshots, or add more photos");
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
        let p = Pool::from_rows(vec![PoolRow { hash: [1; 32], width: 100, height: 100, grid: descriptor::compute(&gray, 100, 100).grid, dhash: 0, faces: vec![Face { rect: [0.3, 0.2, 0.3, 0.3], score: 1.0 }] }], false);
        let mut f = PhotoFilters { people: People::Portraits, colour: Colour::BlackAndWhite, skip_documents: false };
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
}
