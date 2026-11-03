//! Mosaic outline: limits the mosaic to a shape instead of the full rectangle.
//!
//! A tile is placed only when at least `threshold` of its area lies inside
//! the shape; other tiles stay empty and are painted with the background
//! colour. Edges therefore follow tile borders (no photo is cut).
//!
//! The shape is a built-in form or an imported file:
//! - SVG: everything drawn (any non-transparent fill) is inside.
//! - PNG/WebP with transparency: opaque pixels are inside.
//! - Other images (JPEG, PNG without transparency): dark pixels are inside.
//!
//! Imported shapes are centred and scaled to fit inside the artwork. With
//! `cover` they fill it like the motif does (cropping the shape), so a motif
//! with transparency used as its own shape lines up with the picture.

use crate::catalog::hex;
use crate::matching::GridSpec;
use anyhow::{anyhow, bail, Context, Result};
use image::GrayImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Shape {
    Ellipse,
    Heart,
    Hexagon,
    Diamond,
    Star,
    /// Imported from an SVG or raster image.
    File {
        path: PathBuf,
        content_hash: String,
        invert: bool,
        /// Scale to cover the artwork (cropping the shape) instead of fitting it inside.
        #[serde(default)]
        cover: bool,
    },
}

pub const BUILTIN: [&str; 5] = ["ellipse", "heart", "hexagon", "diamond", "star"];

impl Shape {
    /// `name` is a built-in shape or a file path.
    pub fn parse(name: &str, invert: bool) -> Result<Shape> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "ellipse" | "circle" | "oval" => Shape::Ellipse,
            "heart" => Shape::Heart,
            "hexagon" => Shape::Hexagon,
            "diamond" => Shape::Diamond,
            "star" => Shape::Star,
            _ => Shape::from_file(Path::new(name), invert)?,
        })
    }

    pub fn from_file(path: &Path, invert: bool) -> Result<Shape> {
        let bytes = std::fs::read(path).with_context(|| format!("cannot read shape file {}", path.display()))?;
        Ok(Shape::File {
            path: std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            content_hash: hex(blake3::hash(&bytes).as_bytes()),
            invert,
            cover: false,
        })
    }

    pub fn label(&self) -> String {
        match self {
            Shape::Ellipse => "ellipse".into(),
            Shape::Heart => "heart".into(),
            Shape::Hexagon => "hexagon".into(),
            Shape::Diamond => "diamond".into(),
            Shape::Star => "star".into(),
            Shape::File { path, invert, .. } => {
                let n = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if *invert { format!("{n} (inverted)") } else { n }
            }
        }
    }

    /// Inside test for built-in shapes; `x`, `y` in [-1, 1], y pointing down.
    fn inside(&self, x: f64, y: f64) -> bool {
        match self {
            Shape::Ellipse => x * x + y * y <= 1.0,
            Shape::Diamond => x.abs() + y.abs() <= 1.0,
            // Flat top and bottom, points left and right.
            Shape::Hexagon => y.abs() <= 1.0 && x.abs() + 0.5 * y.abs() <= 1.0,
            Shape::Heart => {
                // (X²+Y²−1)³ − X²Y³ ≤ 0 spans X ∈ ±1.139, Y ∈ [−1, 1.236].
                let hx = x * 1.139;
                let hy = -(y * 1.118) + 0.118;
                let a = hx * hx + hy * hy - 1.0;
                a * a * a - hx * hx * hy * hy * hy <= 0.0
            }
            Shape::Star => {
                let pts: Vec<(f64, f64)> = (0..10)
                    .map(|i| {
                        let r = if i % 2 == 0 { 1.0 } else { 0.4 };
                        let a = -std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::PI / 5.0;
                        (r * a.cos(), (r * a.sin() + 0.0955) * 1.105)
                    })
                    .collect();
                point_in_polygon(x, y, &pts)
            }
            Shape::File { .. } => false,
        }
    }
}

fn point_in_polygon(x: f64, y: f64, pts: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let (xi, yi) = pts[i];
        let (xj, yj) = pts[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outline {
    pub shape: Shape,
    /// sRGB colour of the area outside the shape.
    pub background: [u8; 3],
    /// Minimum share of a tile's area inside the shape for it to be placed (0.05–1).
    pub threshold: f32,
}

impl Outline {
    pub fn new(shape: Shape) -> Self {
        Self { shape, background: [255, 255, 255], threshold: 0.5 }
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.05..=1.0).contains(&self.threshold) {
            bail!("shape threshold must be between 0.05 and 1");
        }
        Ok(())
    }

    /// Which tiles (row-major) are part of the mosaic.
    pub fn active_tiles(&self, grid: &GridSpec) -> Result<Vec<bool>> {
        self.validate()?;
        let cov = coverage(&self.shape, grid)?;
        let active: Vec<bool> = cov.iter().map(|&c| c >= self.threshold).collect();
        if !active.iter().any(|&a| a) {
            bail!("the shape covers no tile; use a finer grid, a lower threshold or check the shape file");
        }
        Ok(active)
    }
}


/// Share of each tile's area inside the shape.
pub fn coverage(shape: &Shape, grid: &GridSpec) -> Result<Vec<f32>> {
    let (cols, rows) = (grid.cols as usize, grid.rows as usize);
    // Samples per tile side (fewer for huge grids to bound memory).
    let ss: usize = if grid.tiles() > 250_000 { 2 } else { 6 };
    let (gw, gh) = (cols * ss, rows * ss);
    let samples: Vec<f32> = match shape {
        Shape::File { path, invert, cover, .. } => {
            let m = load_mask(path, *cover, gw as u32, gh as u32)?;
            m.as_raw().iter().map(|&v| if *invert { 1.0 - v as f32 / 255.0 } else { v as f32 / 255.0 }).collect()
        }
        _ => {
            let mut v = Vec::with_capacity(gw * gh);
            for sy in 0..gh {
                for sx in 0..gw {
                    let x = (sx as f64 + 0.5) / gw as f64 * 2.0 - 1.0;
                    let y = (sy as f64 + 0.5) / gh as f64 * 2.0 - 1.0;
                    v.push(if shape.inside(x, y) { 1.0 } else { 0.0 });
                }
            }
            v
        }
    };
    let mut out = vec![0f32; cols * rows];
    for sy in 0..gh {
        for sx in 0..gw {
            out[(sy / ss) * cols + sx / ss] += samples[sy * gw + sx];
        }
    }
    for c in &mut out {
        *c /= (ss * ss) as f32;
    }
    Ok(out)
}

/// Scale and offset that place an `sw × sh` source centred in `w × h`.
fn fit(sw: f64, sh: f64, w: u32, h: u32, cover: bool) -> (f64, f64, f64) {
    let (a, b) = (w as f64 / sw, h as f64 / sh);
    let s = if cover { a.max(b) } else { a.min(b) };
    (s, (w as f64 - sw * s) / 2.0, (h as f64 - sh * s) / 2.0)
}

/// Mask resampled to `w × h` (255 = inside), centred on the artwork.
fn load_mask(path: &Path, cover: bool, w: u32, h: u32) -> Result<GrayImage> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read shape file {}", path.display()))?;
    let is_svg = path.extension().map(|e| e.eq_ignore_ascii_case("svg") || e.eq_ignore_ascii_case("svgz")).unwrap_or(false)
        || looks_like_svg(&bytes);
    if is_svg {
        return render_svg(&bytes, cover, w, h).with_context(|| format!("cannot read SVG shape {}", path.display()));
    }
    let img = image::load_from_memory(&bytes).with_context(|| format!("cannot decode shape image {}", path.display()))?;
    let has_alpha = img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p[3] < 250);
    let gray: GrayImage = if has_alpha {
        let rgba = img.to_rgba8();
        GrayImage::from_raw(rgba.width(), rgba.height(), rgba.pixels().map(|p| p[3]).collect()).unwrap()
    } else {
        let mut l = img.to_luma8();
        for p in l.pixels_mut() {
            p[0] = 255 - p[0];
        }
        l
    };
    // Crop the visible part of the source first (cover), then scale it.
    let (s, ox, oy) = fit(gray.width() as f64, gray.height() as f64, w, h, cover);
    let x0 = ((-ox / s).max(0.0).floor() as u32).min(gray.width() - 1);
    let y0 = ((-oy / s).max(0.0).floor() as u32).min(gray.height() - 1);
    let cw = (((w as f64 - ox.max(0.0) * 2.0) / s).round() as u32).clamp(1, gray.width() - x0);
    let ch = (((h as f64 - oy.max(0.0) * 2.0) / s).round() as u32).clamp(1, gray.height() - y0);
    let sub = image::imageops::crop_imm(&gray, x0, y0, cw, ch).to_image();
    let (rw, rh) = (((cw as f64 * s).round() as u32).clamp(1, w), ((ch as f64 * s).round() as u32).clamp(1, h));
    let scaled = image::imageops::resize(&sub, rw, rh, image::imageops::FilterType::Triangle);
    let mut out = GrayImage::new(w, h);
    image::imageops::overlay(&mut out, &scaled, ((w - rw) / 2) as i64, ((h - rh) / 2) as i64);
    Ok(out)
}

fn looks_like_svg(b: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&b[..b.len().min(1024)]).to_ascii_lowercase();
    head.contains("<svg")
}

fn render_svg(bytes: &[u8], cover: bool, w: u32, h: u32) -> Result<GrayImage> {
    use resvg::{tiny_skia, usvg};
    let tree = usvg::Tree::from_data(bytes, &usvg::Options::default()).map_err(|e| anyhow!("{e}"))?;
    let size = tree.size();
    let (sw, sh) = (size.width(), size.height());
    if !(sw > 0.0 && sh > 0.0) {
        bail!("SVG has no size");
    }
    let (scale, tx, ty) = fit(sw as f64, sh as f64, w, h, cover);
    let (scale, tx, ty) = (scale as f32, tx as f32, ty as f32);
    let mut pm = tiny_skia::Pixmap::new(w, h).ok_or_else(|| anyhow!("mask too large"))?;
    resvg::render(&tree, tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, tx, ty), &mut pm.as_mut());
    Ok(GrayImage::from_raw(w, h, pm.data().chunks_exact(4).map(|p| p[3]).collect()).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(c: u32, r: u32) -> GridSpec {
        GridSpec::new(500.0, 500.0, c, r).unwrap()
    }

    #[test]
    fn ellipse_covers_about_pi_over_four() {
        let cov = coverage(&Shape::Ellipse, &grid(40, 40)).unwrap();
        let share = cov.iter().sum::<f32>() / cov.len() as f32;
        assert!((share - std::f32::consts::FRAC_PI_4).abs() < 0.01, "{share}");
        let a = Outline::new(Shape::Ellipse).active_tiles(&grid(40, 40)).unwrap();
        assert!(!a[0] && a[20 * 40 + 20]);
    }

    #[test]
    fn builtins_are_centred_and_nonempty() {
        for n in BUILTIN {
            let s = Shape::parse(n, false).unwrap();
            let a = Outline::new(s).active_tiles(&grid(30, 30)).unwrap();
            assert!(a[15 * 30 + 15], "{n} centre");
            assert!(!a[0] && !a[29], "{n} corners");
        }
    }

    #[test]
    fn svg_and_png_masks() {
        let dir = std::env::temp_dir().join(format!("mosaic-shape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let svg = dir.join("s.svg");
        std::fs::write(&svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect x="0" y="0" width="50" height="100" fill="red"/></svg>"#).unwrap();
        let a = Outline::new(Shape::from_file(&svg, false).unwrap()).active_tiles(&grid(10, 10)).unwrap();
        assert!(a[0] && a[4] && !a[5] && !a[9]);
        let inv = Outline::new(Shape::from_file(&svg, true).unwrap()).active_tiles(&grid(10, 10)).unwrap();
        assert!(!inv[0] && inv[9]);

        // Opaque image: dark = inside (bottom half black).
        let png = dir.join("m.png");
        let img = image::GrayImage::from_fn(20, 20, |_, y| image::Luma([if y >= 10 { 0 } else { 255 }]));
        img.save(&png).unwrap();
        let a = Outline::new(Shape::from_file(&png, false).unwrap()).active_tiles(&grid(10, 10)).unwrap();
        assert!(!a[0] && a[99]);

        // Transparency: opaque = inside (left column opaque white).
        let png2 = dir.join("a.png");
        let img = image::RgbaImage::from_fn(20, 20, |x, _| image::Rgba([255, 255, 255, if x < 10 { 255 } else { 0 }]));
        img.save(&png2).unwrap();
        let a = Outline::new(Shape::from_file(&png2, false).unwrap()).active_tiles(&grid(10, 10)).unwrap();
        assert!(a[0] && !a[9]);
    }

    #[test]
    fn file_shapes_fit_inside_unless_cover() {
        let dir = std::env::temp_dir().join(format!("mosaic-shape-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let svg = dir.join("c.svg");
        std::fs::write(&svg, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="50"/></svg>"#).unwrap();
        let g = GridSpec::new(1000.0, 500.0, 20, 10).unwrap();
        let mut o = Outline::new(Shape::from_file(&svg, false).unwrap());
        let a = o.active_tiles(&g).unwrap();
        assert!(a[10] && a[9 * 20 + 10], "top and bottom centre inside");
        assert!(!a[5 * 20] && !a[5 * 20 + 19], "left and right edges outside");
        if let Shape::File { cover, .. } = &mut o.shape {
            *cover = true;
        }
        let a = o.active_tiles(&g).unwrap();
        assert!(a[5 * 20 + 1] && a[5 * 20 + 18] && !a[0], "cover fills the width");
    }
}
