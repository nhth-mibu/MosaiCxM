//! Offline source-photo filters. Face boxes and colour grids come from the
//! index; document detection uses cached thumbnails and screenshot filenames.

use crate::catalog::{Catalog, Hash};
use crate::color::dequantize_lab;
use crate::decode;
use crate::job::Job;
use crate::pool::{Pool, PoolPhoto};
use anyhow::{bail, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
        let screenshots = if self.skip_documents { screenshot_hashes(cat, libs)? } else { HashSet::new() };
        let mut photos = Vec::new();
        for (i, p) in pool.photos.iter().enumerate() {
            if i % 128 == 0 {
                job.check()?;
                job.report("filtering source photos", i as u64, pool.photos.len() as u64);
            }
            if !self.allows(p) { continue; }
            if self.skip_documents {
                if screenshots.contains(&p.hash) { continue; }
                let cached = cat.conn.query_row("SELECT document FROM photo_classification WHERE content_hash = ?1 AND version = 1", [&p.hash[..]], |r| r.get::<_, bool>(0));
                let document = match cached {
                    Ok(v) => v,
                    Err(rusqlite::Error::QueryReturnedNoRows) => {
                        let bytes = cat.thumb(&p.hash)?.ok_or_else(|| anyhow::anyhow!("a photo thumbnail is missing; scan your folders again"))?;
                        let document = likely_document(&decode::decode_bytes(&bytes, None)?.image);
                        cat.conn.execute("INSERT OR REPLACE INTO photo_classification(content_hash, version, document) VALUES(?1, 1, ?2)", rusqlite::params![&p.hash[..], document])?;
                        document
                    }
                    Err(e) => return Err(e.into()),
                };
                // A substantial face is evidence of a portrait, even against
                // a plain white backdrop. Filename-marked screenshots still go.
                if document && !p.faces.iter().any(|f| f.rect[2] * f.rect[3] >= 0.02) { continue; }
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
    let small = image::imageops::thumbnail(img, 192, 192);
    let (w, h) = small.dimensions();
    if w < 24 || h < 24 { return false; }
    let distance = |a: image::Rgb<u8>, b: image::Rgb<u8>| (0..3).map(|k| a[k].abs_diff(b[k]) as u32).sum::<u32>();
    let mut flat = 0u32;
    let mut text_rows = 0u32;
    let mut edges = 0u32;
    for y in 1..h {
        let mut transitions = 0;
        for x in 1..w {
            let p = *small.get_pixel(x, y);
            let left = *small.get_pixel(x - 1, y);
            let above = *small.get_pixel(x, y - 1);
            if distance(p, left) < 18 && distance(p, above) < 18 { flat += 1; }
            if distance(p, left) > 150 { transitions += 1; edges += 1; }
        }
        if transitions >= 8 && transitions < w / 2 { text_rows += 1; }
    }
    let n = ((w - 1) * (h - 1)) as f32;
    flat as f32 / n > 0.62 && text_rows as f32 / h as f32 > 0.12 && edges as f32 / n > 0.015
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{catalog::PoolRow, descriptor, faces::Face};

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
        let thumb = image::imageops::thumbnail(&page, 96, 96);
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 85).encode(thumb.as_raw(), 96, 96, jpeg_encoder::ColorType::Rgb).unwrap();
        assert!(likely_document(&decode::decode_bytes(&jpeg, None).unwrap().image), "must also work on stored JPEG thumbnails");
        assert!(!likely_document(&RgbImage::from_fn(192, 192, |x, y| image::Rgb([x as u8, y as u8, 100]))));
    }
}
