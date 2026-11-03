//! Region renderer. Any rectangle of the global artwork raster can be rendered
//! independently (strips for BigTIFF, pages for PDF, the whole thing for a
//! preview). Each mosaic tile is always resampled at its full global pixel
//! size and then cut, so neighbouring regions share identical pixels and page
//! boundaries cannot introduce seams.

use crate::catalog::{Catalog, Hash};
use crate::color::{lab_to_srgb8, linear_rgb_to_lab, linear_to_srgb8, srgb8_to_linear, Lab};
use crate::decode;
use crate::descriptor::CELLS;
use crate::job::Job;
use crate::project::Project;
use crate::units::{edge, RasterSpec};
use anyhow::{anyhow, Result};
use image::RgbImage;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Supplies upright source images by source index.
pub trait TileSource: Sync {
    /// Image with a short side of at least `min_short_side` where possible.
    fn load(&self, source: u32, min_short_side: u32) -> Result<RgbImage>;
    fn name(&self) -> &'static str;
}

/// Stored thumbnails (previews; never touches original files).
pub struct ThumbSource {
    thumbs: Vec<Vec<u8>>,
}

impl ThumbSource {
    pub fn load_for(cat: &Catalog, hashes: &[Hash]) -> Result<Self> {
        let thumbs = hashes
            .iter()
            .map(|h| cat.thumb(h)?.ok_or_else(|| anyhow!("thumbnail missing for {}; update the library index", crate::catalog::hex(h))))
            .collect::<Result<_>>()?;
        Ok(Self { thumbs })
    }
}

impl TileSource for ThumbSource {
    fn load(&self, source: u32, _min: u32) -> Result<RgbImage> {
        Ok(decode::decode_bytes(&self.thumbs[source as usize], None)?.image)
    }
    fn name(&self) -> &'static str {
        "thumbnails"
    }
}

/// Original files, decoded at the needed resolution.
pub struct FileSource {
    pub paths: Vec<PathBuf>,
}

impl TileSource for FileSource {
    fn load(&self, source: u32, min: u32) -> Result<RgbImage> {
        let p = &self.paths[source as usize];
        Ok(decode::decode_file(p, Some(min)).map_err(|e| anyhow!("{}: {e}", p.display()))?.image)
    }
    fn name(&self) -> &'static str {
        "original files"
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct PixKey {
    photo: u32,
    crop: [u32; 4],
    adjust: [u32; 3],
    w: u32,
    h: u32,
}

struct Cache {
    map: HashMap<PixKey, (Arc<RgbImage>, u64)>,
    bytes: u64,
    budget: u64,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl Cache {
    fn get(&mut self, k: &PixKey) -> Option<Arc<RgbImage>> {
        self.clock += 1;
        let c = self.clock;
        match self.map.get_mut(k) {
            Some(e) => {
                e.1 = c;
                self.hits += 1;
                Some(e.0.clone())
            }
            None => None,
        }
    }
    fn put(&mut self, k: PixKey, v: Arc<RgbImage>) {
        self.clock += 1;
        let sz = v.as_raw().len() as u64;
        if let Some(old) = self.map.insert(k, (v, self.clock)) {
            self.bytes -= old.0.as_raw().len() as u64;
        }
        self.bytes += sz;
        if self.bytes > self.budget {
            // Evict least recently used entries down to 80% of the budget.
            let mut entries: Vec<(u64, PixKey)> = self.map.iter().map(|(k, v)| (v.1, *k)).collect();
            entries.sort_unstable_by_key(|e| e.0);
            for (_, k) in entries {
                if self.bytes <= self.budget * 8 / 10 {
                    break;
                }
                if let Some(v) = self.map.remove(&k) {
                    self.bytes -= v.0.as_raw().len() as u64;
                }
            }
        }
    }
}

pub struct Renderer<'a> {
    pub spec: RasterSpec,
    project: &'a Project,
    source: &'a dyn TileSource,
    xs: Vec<u64>,
    ys: Vec<u64>,
    cache: Mutex<Cache>,
    blend: Option<(f32, Vec<[u8; 3]>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_peak_bytes: u64,
}

impl<'a> Renderer<'a> {
    pub fn new(project: &'a Project, spec: RasterSpec, source: &'a dyn TileSource, cache_budget: u64) -> Result<Self> {
        let g = &project.grid;
        let xs = (0..=g.cols as u64).map(|i| edge(i, g.cols as u64, spec.width_px)).collect();
        let ys = (0..=g.rows as u64).map(|i| edge(i, g.rows as u64, spec.height_px)).collect();
        let blend = if project.settings.motif_blend > 0.0 {
            Some((project.settings.motif_blend, project.target_cells()?.into_iter().map(lab_to_srgb8).collect()))
        } else {
            None
        };
        Ok(Self {
            spec,
            project,
            source,
            xs,
            ys,
            cache: Mutex::new(Cache { map: HashMap::new(), bytes: 0, budget: cache_budget, clock: 0, hits: 0, misses: 0 }),
            blend,
        })
    }

    pub fn stats(&self) -> RenderStats {
        let c = self.cache.lock().unwrap();
        RenderStats { cache_hits: c.hits, cache_misses: c.misses, cache_peak_bytes: c.bytes }
    }

    /// Colour shown where no photo is placed.
    pub fn background(&self) -> [u8; 3] {
        self.project.outline.as_ref().map(|o| o.background).unwrap_or([255; 3])
    }

    /// Largest tile pixmap in bytes (for memory estimates).
    pub fn max_tile_bytes(&self) -> u64 {
        let mw = self.xs.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let mh = self.ys.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        mw * mh * 3
    }

    pub fn tile_rect(&self, i: usize) -> (u64, u64, u64, u64) {
        let c = self.project.grid.cols as usize;
        let (tx, ty) = (i % c, i / c);
        (self.xs[tx], self.ys[ty], self.xs[tx + 1], self.ys[ty + 1])
    }

    fn key(&self, i: usize) -> PixKey {
        let t = &self.project.tiles[i];
        let (x0, y0, x1, y1) = self.tile_rect(i);
        PixKey {
            photo: t.photo,
            crop: t.crop.map(f32::to_bits),
            adjust: t.adjust.map(f32::to_bits),
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        }
    }

    fn make_pixmap(&self, i: usize) -> Result<RgbImage> {
        let t = &self.project.tiles[i];
        let k = self.key(i);
        let s = &self.project.sources[t.photo as usize];
        // Decode just large enough that the crop covers the tile at 1:1 or better.
        let scale = (k.w as f64 / (t.crop[2] as f64 * s.width as f64)).max(k.h as f64 / (t.crop[3] as f64 * s.height as f64));
        let need = (scale * s.width.min(s.height) as f64).ceil() as u32 + 2;
        let src = self.source.load(t.photo, need.max(8))?;
        let mut img = decode::resize_crop(&src, t.crop, k.w.max(1), k.h.max(1))?;
        if t.adjust != [0.0; 3] {
            apply_lab_offset(&mut img, t.adjust);
        }
        Ok(img)
    }

    fn tiles_in(&self, x0: u64, y0: u64, x1: u64, y1: u64) -> Vec<usize> {
        let c0 = self.xs.partition_point(|&e| e <= x0).saturating_sub(1);
        let c1 = self.xs.partition_point(|&e| e < x1).min(self.xs.len() - 1);
        let r0 = self.ys.partition_point(|&e| e <= y0).saturating_sub(1);
        let r1 = self.ys.partition_point(|&e| e < y1).min(self.ys.len() - 1);
        let cols = self.project.grid.cols as usize;
        let mut v = Vec::new();
        for r in r0..r1 {
            for c in c0..c1 {
                v.push(r * cols + c);
            }
        }
        v
    }

    /// Render global pixel rectangle [x0, x0+w) × [y0, y0+h) as RGB8.
    pub fn render_region(&self, x0: u64, y0: u64, w: u64, h: u64, job: &Job) -> Result<Vec<u8>> {
        let len = w.checked_mul(h).and_then(|v| v.checked_mul(3)).ok_or_else(|| anyhow!("region too large"))?;
        let bg = self.background();
        let mut buf: Vec<u8> = bg.iter().copied().cycle().take(usize::try_from(len)?).collect();
        let (x1, y1) = (x0 + w, y0 + h);
        let all = self.tiles_in(x0, y0, x1, y1);
        let tiles: Vec<usize> = all.iter().copied().filter(|&i| !self.project.tiles[i].is_empty()).collect();
        let has_empty = tiles.len() != all.len();
        // Render missing pixmaps in parallel, in bounded batches.
        for batch in tiles.chunks(256) {
            job.check()?;
            let mut todo: Vec<(usize, PixKey)> = Vec::new();
            {
                let mut c = self.cache.lock().unwrap();
                let mut seen = std::collections::HashSet::new();
                for &i in batch {
                    let k = self.key(i);
                    if c.map.contains_key(&k) || !seen.insert(k) {
                        continue;
                    }
                    c.misses += 1;
                    todo.push((i, k));
                }
            }
            let made: Vec<(PixKey, Result<RgbImage>)> = todo.par_iter().map(|(i, k)| (*k, self.make_pixmap(*i))).collect();
            let mut pix: HashMap<PixKey, Arc<RgbImage>> = HashMap::new();
            for (k, r) in made {
                pix.insert(k, Arc::new(r?));
            }
            for &i in batch {
                let k = self.key(i);
                let p = match pix.get(&k) {
                    Some(p) => p.clone(),
                    None => self.cache.lock().unwrap().get(&k).ok_or_else(|| anyhow!("cache miss"))?,
                };
                let (tx0, ty0, tx1, ty1) = self.tile_rect(i);
                let (ix0, iy0, ix1, iy1) = (tx0.max(x0), ty0.max(y0), tx1.min(x1), ty1.min(y1));
                if ix0 >= ix1 || iy0 >= iy1 {
                    continue;
                }
                let tw = (tx1 - tx0) as usize;
                let n = (ix1 - ix0) as usize * 3;
                for y in iy0..iy1 {
                    let src = ((y - ty0) as usize * tw + (ix0 - tx0) as usize) * 3;
                    let dst = ((y - y0) as usize * w as usize + (ix0 - x0) as usize) * 3;
                    buf[dst..dst + n].copy_from_slice(&p.as_raw()[src..src + n]);
                }
            }
            let mut c = self.cache.lock().unwrap();
            for (k, v) in pix {
                c.put(k, v);
            }
        }
        if let Some((alpha, cells)) = &self.blend {
            self.apply_blend(&mut buf, x0, y0, w, h, *alpha, cells);
            if has_empty {
                // The motif is not blended outside the outline.
                for &i in all.iter().filter(|&&i| self.project.tiles[i].is_empty()) {
                    let (tx0, ty0, tx1, ty1) = self.tile_rect(i);
                    let (ix0, iy0, ix1, iy1) = (tx0.max(x0), ty0.max(y0), tx1.min(x1), ty1.min(y1));
                    for y in iy0..iy1 {
                        let row = (y - y0) as usize * w as usize;
                        for x in ix0..ix1 {
                            let o = (row + (x - x0) as usize) * 3;
                            buf[o..o + 3].copy_from_slice(&bg);
                        }
                    }
                }
            }
        }
        Ok(buf)
    }

    /// Blend the motif (bilinear over the target cell grid) at global coordinates.
    #[allow(clippy::too_many_arguments)]
    fn apply_blend(&self, buf: &mut [u8], x0: u64, y0: u64, w: u64, _h: u64, alpha: f32, cells: &[[u8; 3]]) {
        let gw = self.project.grid.cols as usize * CELLS;
        let gh = self.project.grid.rows as usize * CELLS;
        let sx = gw as f64 / self.spec.width_px as f64;
        let sy = gh as f64 / self.spec.height_px as f64;
        buf.par_chunks_mut(w as usize * 3).enumerate().for_each(|(ry, row)| {
            let gy = ((y0 + ry as u64) as f64 + 0.5) * sy - 0.5;
            let (cy0, fy) = (gy.floor().clamp(0.0, (gh - 1) as f64) as usize, (gy - gy.floor()).clamp(0.0, 1.0) as f32);
            let cy1 = (cy0 + 1).min(gh - 1);
            for rx in 0..w as usize {
                let gx = ((x0 + rx as u64) as f64 + 0.5) * sx - 0.5;
                let (cx0, fx) = (gx.floor().clamp(0.0, (gw - 1) as f64) as usize, (gx - gx.floor()).clamp(0.0, 1.0) as f32);
                let cx1 = (cx0 + 1).min(gw - 1);
                for ch in 0..3 {
                    let a = cells[cy0 * gw + cx0][ch] as f32 * (1.0 - fx) + cells[cy0 * gw + cx1][ch] as f32 * fx;
                    let b = cells[cy1 * gw + cx0][ch] as f32 * (1.0 - fx) + cells[cy1 * gw + cx1][ch] as f32 * fx;
                    let m = a * (1.0 - fy) + b * fy;
                    let p = &mut row[rx * 3 + ch];
                    *p = (*p as f32 * (1.0 - alpha) + m * alpha).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
    }
}

fn apply_lab_offset(img: &mut RgbImage, off: Lab) {
    for p in img.pixels_mut() {
        let lab = linear_rgb_to_lab(srgb8_to_linear(p[0]), srgb8_to_linear(p[1]), srgb8_to_linear(p[2]));
        let out = lab_to_srgb8([lab[0] + off[0], lab[1] + off[1], lab[2] + off[2]]);
        p.0 = out;
    }
    let _ = linear_to_srgb8;
}

/// Render a whole-mosaic preview from thumbnails.
pub fn render_preview(cat: &Catalog, project: &Project, tile_px: u32, job: &Job) -> Result<RgbImage> {
    let g = &project.grid;
    let (tw, th) = g.tile_mm();
    let tile_px = tile_px.clamp(2, 256);
    // Keep the tile's physical aspect: tile_px is the long side.
    let (pw, ph) = if tw >= th { (tile_px, ((tile_px as f64 * th / tw).round() as u32).max(1)) } else { (((tile_px as f64 * tw / th).round() as u32).max(1), tile_px) };
    let w = g.cols as u64 * pw as u64;
    let h = g.rows as u64 * ph as u64;
    if w * h > 400_000_000 {
        anyhow::bail!("preview of {w}×{h} px is too large; use a smaller tile size");
    }
    let spec = RasterSpec::with_pixels(g.width_mm, g.height_mm, w, h)?;
    let thumbs = ThumbSource::load_for(cat, &project.source_hashes()?)?;
    let r = Renderer::new(project, spec, &thumbs, 512 << 20)?;
    let mut img = RgbImage::new(w as u32, h as u32);
    let band = 512u64.max(ph as u64);
    let mut y = 0;
    while y < h {
        let bh = band.min(h - y);
        let buf = r.render_region(0, y, w, bh, job)?;
        let off = (y * w * 3) as usize;
        img.as_mut()[off..off + buf.len()].copy_from_slice(&buf);
        y += bh;
        job.report("rendering preview", y, h);
    }
    Ok(img)
}
