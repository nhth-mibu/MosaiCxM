//! Zoomed-in preview. The normal preview is built from small thumbnails, so it
//! goes blurry when you zoom in. When the view stops moving, the visible part
//! is rendered again at screen resolution. Tiles that need more detail than a
//! thumbnail has are decoded from the original photo files.

use anyhow::{anyhow, Result};
use image::RgbImage;
use mosaic_core::catalog::{unhex, Catalog};
use mosaic_core::decode;
use mosaic_core::descriptor::THUMB_SHORT_SIDE;
use mosaic_core::job::Job;
use mosaic_core::project::{check_sources, Project};
use mosaic_core::render::{Renderer, TileSource};
use mosaic_core::units::{RasterSpec, MAX_RASTER_SIDE};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// Largest tile size in screen pixels that is worth rendering.
const MAX_TILE_PX: f64 = 1600.0;
/// Extra area rendered around the visible part, so small moves stay sharp.
const MARGIN: f64 = 0.15;
/// Memory for decoded photos kept between renders.
const PHOTO_CACHE_BYTES: u64 = 768 << 20;

/// What the preview shows, in logical pixels, as sent by the UI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub cx: f64,
    pub cy: f64,
    /// Size of the whole mosaic on screen at the current zoom.
    pub img_w: f64,
    pub img_h: f64,
    /// Size of the preview area.
    pub view_w: f64,
    pub view_h: f64,
    pub scale: f64,
}

/// A region of a virtual raster of `w` × `h` pixels to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub w: u64,
    pub h: u64,
    pub x0: u64,
    pub y0: u64,
    pub x1: u64,
    pub y1: u64,
}

impl Plan {
    /// The rendered region as fractions of the mosaic: (u0, v0, u1, v1).
    pub fn uv(&self) -> [f32; 4] {
        [self.x0 as f32 / self.w as f32, self.y0 as f32 / self.h as f32, self.x1 as f32 / self.w as f32, self.y1 as f32 / self.h as f32]
    }
}

/// Works out what to render for `view`, or `None` when the normal preview
/// (`base_w` pixels wide) is already sharp enough.
pub fn plan(cols: u32, view: &View, base_w: u32) -> Option<Plan> {
    let (mut w, mut h) = (view.img_w * view.scale, view.img_h * view.scale);
    if !(w > 0.0 && h > 0.0) || w <= base_w as f64 * 1.1 {
        return None;
    }
    // Past this the photos cannot show more detail.
    let k = (MAX_TILE_PX * cols.max(1) as f64 / w).min(MAX_RASTER_SIDE as f64 / w.max(h)).min(1.0);
    w *= k;
    h *= k;
    let hu = view.view_w / view.img_w / 2.0 * (1.0 + MARGIN);
    let hv = view.view_h / view.img_h / 2.0 * (1.0 + MARGIN);
    let (u0, u1) = ((view.cx - hu).max(0.0), (view.cx + hu).min(1.0));
    let (v0, v1) = ((view.cy - hv).max(0.0), (view.cy + hv).min(1.0));
    let (w, h) = (w.round().max(1.0) as u64, h.round().max(1.0) as u64);
    let x0 = ((u0 * w as f64).floor() as u64).min(w - 1);
    let y0 = ((v0 * h as f64).floor() as u64).min(h - 1);
    let x1 = ((u1 * w as f64).ceil() as u64).clamp(x0 + 1, w);
    let y1 = ((v1 * h as f64).ceil() as u64).clamp(y0 + 1, h);
    Some(Plan { w, h, x0, y0, x1, y1 })
}

/// Index of the tile at (u, v), fractions of the mosaic, if a photo is there.
pub fn tile_at(p: &Project, u: f64, v: f64) -> Option<usize> {
    if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
        return None;
    }
    let (cols, rows) = (p.grid.cols as usize, p.grid.rows as usize);
    let i = ((v * rows as f64) as usize).min(rows - 1) * cols + ((u * cols as f64) as usize).min(cols - 1);
    p.tiles.get(i).filter(|t| !t.is_empty()).map(|_| i)
}

struct Cached {
    img: Arc<RgbImage>,
    /// True when no larger version can be had (full original, or only a thumbnail).
    best: bool,
    used: u64,
}

#[derive(Default)]
struct PhotoCache {
    map: HashMap<u32, Cached>,
    bytes: u64,
    clock: u64,
}

impl PhotoCache {
    fn get(&mut self, photo: u32, min: u32) -> Option<Arc<RgbImage>> {
        self.clock += 1;
        let e = self.map.get_mut(&photo)?;
        if e.best || e.img.width().min(e.img.height()) >= min {
            e.used = self.clock;
            Some(e.img.clone())
        } else {
            None
        }
    }
    fn put(&mut self, photo: u32, img: Arc<RgbImage>, best: bool) {
        self.clock += 1;
        self.bytes += img.as_raw().len() as u64;
        if let Some(old) = self.map.insert(photo, Cached { img, best, used: self.clock }) {
            self.bytes -= old.img.as_raw().len() as u64;
        }
        if self.bytes > PHOTO_CACHE_BYTES {
            let mut order: Vec<(u64, u32)> = self.map.iter().map(|(k, v)| (v.used, *k)).collect();
            order.sort_unstable();
            for (_, k) in order {
                if self.bytes <= PHOTO_CACHE_BYTES * 8 / 10 {
                    break;
                }
                if let Some(v) = self.map.remove(&k) {
                    self.bytes -= v.img.as_raw().len() as u64;
                }
            }
        }
    }
}

/// Photos for one project: thumbnails where they are enough, original files
/// where more detail is needed. Decoded photos are kept between renders.
pub struct ZoomSource {
    pub project: Arc<Project>,
    data_dir: PathBuf,
    paths: OnceLock<Vec<Option<PathBuf>>>,
    cat: Mutex<Option<Catalog>>,
    cache: Mutex<PhotoCache>,
}

impl ZoomSource {
    pub fn new(project: Arc<Project>, data_dir: PathBuf) -> Self {
        Self { project, data_dir, paths: OnceLock::new(), cat: Mutex::new(None), cache: Mutex::default() }
    }

    /// Where each source photo is on disk (looked up once; missing ones are `None`).
    pub fn paths(&self) -> &[Option<PathBuf>] {
        self.paths.get_or_init(|| self.with_catalog(|cat| Ok(check_sources(cat, &self.project, &Job::default())?.resolved)).unwrap_or_default())
    }

    /// File path of a source photo, if it was already looked up.
    pub fn known_path(&self, source: usize) -> Option<&PathBuf> {
        self.paths.get()?.get(source)?.as_ref()
    }

    fn with_catalog<T>(&self, f: impl FnOnce(&Catalog) -> Result<T>) -> Result<T> {
        let mut c = self.cat.lock().unwrap();
        if c.is_none() {
            *c = Some(Catalog::open(&self.data_dir)?);
        }
        f(c.as_ref().unwrap())
    }

    fn thumbnail(&self, source: u32) -> Result<RgbImage> {
        let s = self.project.sources.get(source as usize).ok_or_else(|| anyhow!("no source photo {source}"))?;
        let h = unhex(&s.hash)?;
        let bytes = self.with_catalog(|cat| cat.thumb(&h)?.ok_or_else(|| anyhow!("thumbnail missing")))?;
        Ok(decode::decode_bytes(&bytes, None)?.image)
    }

    /// Renders one region of the mosaic.
    pub fn render(&self, plan: &Plan, job: &Job) -> Result<(Vec<u8>, u32, u32)> {
        let g = &self.project.grid;
        let spec = RasterSpec::with_pixels(g.width_mm, g.height_mm, plan.w, plan.h)?;
        let r = Renderer::new(&self.project, spec, self, 256 << 20)?;
        let (w, h) = (plan.x1 - plan.x0, plan.y1 - plan.y0);
        Ok((r.render_region(plan.x0, plan.y0, w, h, job)?, w as u32, h as u32))
    }
}

impl TileSource for ZoomSource {
    fn load(&self, source: u32, min: u32) -> Result<RgbImage> {
        if let Some(img) = self.cache.lock().unwrap().get(source, min) {
            return Ok((*img).clone());
        }
        let original = if min > THUMB_SHORT_SIDE {
            self.paths().get(source as usize).and_then(|p| p.as_ref()).and_then(|p| decode::decode_file(p, Some(min)).ok())
        } else {
            None
        };
        let (img, best) = match original {
            Some(d) => {
                let best = d.image.width().min(d.image.height()) < min;
                (d.image, best)
            }
            // The original is missing or unreadable: the thumbnail is all there is.
            None => (self.thumbnail(source)?, min > THUMB_SHORT_SIDE),
        };
        self.cache.lock().unwrap().put(source, Arc::new(img.clone()), best);
        Ok(img)
    }
    fn name(&self) -> &'static str {
        "zoom"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(zoom: f64) -> View {
        View { cx: 0.5, cy: 0.5, img_w: 800.0 * zoom, img_h: 600.0 * zoom, view_w: 800.0, view_h: 600.0, scale: 1.0 }
    }

    #[test]
    fn no_detail_while_preview_is_sharp() {
        assert_eq!(plan(60, &view(1.0), 1440), None);
        assert_eq!(plan(60, &view(1.5), 1440), None);
    }

    #[test]
    fn renders_visible_part_with_margin() {
        let p = plan(60, &view(4.0), 1440).unwrap();
        assert_eq!((p.w, p.h), (3200, 2400));
        // A quarter of the width is visible, plus the margin on both sides.
        let [u0, v0, u1, v1] = p.uv();
        assert!((u1 - u0 - 0.25 * 1.15).abs() < 0.01, "{u0} {u1}");
        assert!((v1 - v0 - 0.25 * 1.15).abs() < 0.01, "{v0} {v1}");
        assert!((u0 + u1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn tile_size_is_capped() {
        // 10 columns at zoom 100: 8000 px per tile, capped to MAX_TILE_PX.
        let p = plan(10, &view(100.0), 600).unwrap();
        assert_eq!(p.w, (MAX_TILE_PX * 10.0) as u64);
        assert!(p.x1 > p.x0 && p.y1 > p.y0);
    }

    #[test]
    fn edge_of_mosaic_is_clamped() {
        let mut v = view(4.0);
        v.cx = 0.0;
        let p = plan(60, &v, 1440).unwrap();
        assert_eq!(p.x0, 0);
    }
}
