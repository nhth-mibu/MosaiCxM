//! Compact perceptual descriptors.
//!
//! Each photo is stored as a 16×16 grid of quantised CIELAB cells covering the
//! whole upright image (768 bytes) plus a 64-bit difference hash. Because the
//! grid covers the uncropped frame, a descriptor for *any* crop (tile aspect
//! ratio and crop position) can be derived later without decoding the photo
//! again. Matching uses 4×4 cells per tile (48 numbers).

use crate::color::{dequantize_lab, linear_rgb_to_lab, quantize_lab, srgb8_to_linear, Lab};
use image::RgbImage;

/// Bump when extraction logic changes; cached descriptors with another
/// version are recomputed on the next index update.
pub const DESCRIPTOR_VERSION: u32 = 1;
pub const GRID: usize = 16;
pub const GRID_BYTES: usize = GRID * GRID * 3;
pub const CELLS: usize = 4;
pub const DESC_DIM: usize = CELLS * CELLS * 3;
/// Decode hint (short side, px) used for analysis.
pub const ANALYSIS_SHORT_SIDE: u32 = 128;
/// Stored thumbnail short side (px) used for previews.
pub const THUMB_SHORT_SIDE: u32 = 96;

pub type TileDesc = [f32; DESC_DIM];

#[derive(Debug, Clone)]
pub struct Descriptor {
    pub width: u32,
    pub height: u32,
    pub grid: Vec<u8>,
    pub dhash: u64,
}

/// Area-average an image into `gw × gh` cells in linear light, returned as Lab.
pub fn area_average_lab(img: &RgbImage, gw: usize, gh: usize) -> Vec<Lab> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut acc = vec![[0f64; 4]; gw * gh];
    // Each pixel contributes to cells with fractional coverage along each axis.
    let wx = axis_weights(w, gw);
    let wy = axis_weights(h, gh);
    let raw = img.as_raw();
    for y in 0..h {
        for &(cy, fy) in &wy[y] {
            let row = &mut acc[cy * gw..(cy + 1) * gw];
            for x in 0..w {
                let p = &raw[(y * w + x) * 3..(y * w + x) * 3 + 3];
                let lin = [srgb8_to_linear(p[0]), srgb8_to_linear(p[1]), srgb8_to_linear(p[2])];
                for &(cx, fx) in &wx[x] {
                    let f = fx * fy;
                    let a = &mut row[cx];
                    a[0] += lin[0] as f64 * f;
                    a[1] += lin[1] as f64 * f;
                    a[2] += lin[2] as f64 * f;
                    a[3] += f;
                }
            }
        }
    }
    acc.iter()
        .map(|a| {
            let n = a[3].max(1e-12);
            linear_rgb_to_lab((a[0] / n) as f32, (a[1] / n) as f32, (a[2] / n) as f32)
        })
        .collect()
}

/// For each of `n` pixels, the (cell, weight) pairs it overlaps when `n`
/// pixels are split into `cells` equal cells.
fn axis_weights(n: usize, cells: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = cells as f64 / n as f64;
    (0..n)
        .map(|i| {
            let a = i as f64 * scale;
            let b = (i + 1) as f64 * scale;
            let mut v = Vec::with_capacity(2);
            let mut c = a.floor() as usize;
            while (c as f64) < b && c < cells {
                let lo = a.max(c as f64);
                let hi = b.min(c as f64 + 1.0);
                if hi > lo {
                    v.push((c, hi - lo));
                }
                c += 1;
            }
            v
        })
        .collect()
}

pub fn compute(img: &RgbImage, full_width: u32, full_height: u32) -> Descriptor {
    let cells = area_average_lab(img, GRID, GRID);
    let mut grid = Vec::with_capacity(GRID_BYTES);
    for c in &cells {
        grid.extend_from_slice(&quantize_lab(*c));
    }
    let dh = area_average_lab(img, 9, 8);
    let mut dhash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            if dh[y * 9 + x + 1][0] > dh[y * 9 + x][0] {
                dhash |= 1 << (y * 8 + x);
            }
        }
    }
    Descriptor { width: full_width, height: full_height, grid, dhash }
}

/// Normalised crop rectangle [x, y, w, h] of a photo with aspect `photo_aspect`
/// (w/h) that fills a tile with aspect `tile_aspect`. `t` in [0,1] slides the
/// crop along the long axis (0.5 = centre).
pub fn crop_rect(photo_aspect: f32, tile_aspect: f32, t: f32) -> [f32; 4] {
    if photo_aspect > tile_aspect {
        let fw = tile_aspect / photo_aspect;
        [(1.0 - fw) * t, 0.0, fw, 1.0]
    } else {
        let fh = photo_aspect / tile_aspect;
        [0.0, (1.0 - fh) * t, 1.0, fh]
    }
}

/// Crop positions considered for a photo. Centre only unless the crop discards
/// a substantial part of the frame.
pub fn crop_positions(photo_aspect: f32, tile_aspect: f32, allow_offsets: bool) -> &'static [f32] {
    let r = crop_rect(photo_aspect, tile_aspect, 0.5);
    let kept = r[2] * r[3];
    if allow_offsets && kept < 0.8 { &[0.5, 0.0, 1.0] } else { &[0.5] }
}

/// Derive the 4×4 Lab tile descriptor of a crop from a stored 16×16 grid.
pub fn crop_descriptor(grid: &[u8], crop: [f32; 4]) -> TileDesc {
    let wx = crop_axis_weights(crop[0], crop[2]);
    let wy = crop_axis_weights(crop[1], crop[3]);
    let mut out = [0f32; DESC_DIM];
    for cy in 0..CELLS {
        for cx in 0..CELLS {
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for gy in 0..GRID {
                let fy = wy[cy][gy];
                if fy == 0.0 {
                    continue;
                }
                for gx in 0..GRID {
                    let f = fy * wx[cx][gx];
                    if f == 0.0 {
                        continue;
                    }
                    let i = (gy * GRID + gx) * 3;
                    let lab = dequantize_lab([grid[i], grid[i + 1], grid[i + 2]]);
                    acc[0] += lab[0] * f;
                    acc[1] += lab[1] * f;
                    acc[2] += lab[2] * f;
                    wsum += f;
                }
            }
            let o = (cy * CELLS + cx) * 3;
            for k in 0..3 {
                out[o + k] = acc[k] / wsum.max(1e-9);
            }
        }
    }
    out
}

fn crop_axis_weights(start: f32, len: f32) -> [[f32; GRID]; CELLS] {
    let mut w = [[0f32; GRID]; CELLS];
    for (c, row) in w.iter_mut().enumerate() {
        let a = (start + len * c as f32 / CELLS as f32) * GRID as f32;
        let b = (start + len * (c + 1) as f32 / CELLS as f32) * GRID as f32;
        for (g, v) in row.iter_mut().enumerate() {
            let lo = a.max(g as f32);
            let hi = b.min(g as f32 + 1.0);
            if hi > lo {
                *v = hi - lo;
            }
        }
    }
    w
}

pub fn mean_lab(d: &TileDesc) -> Lab {
    let mut m = [0f32; 3];
    for c in d.chunks_exact(3) {
        m[0] += c[0];
        m[1] += c[1];
        m[2] += c[2];
    }
    let n = (CELLS * CELLS) as f32;
    [m[0] / n, m[1] / n, m[2] / n]
}

/// 2×2 Lab summary (12 numbers) used as the retrieval key.
pub fn key12(d: &TileDesc) -> [f32; 12] {
    let mut k = [0f32; 12];
    for cy in 0..CELLS {
        for cx in 0..CELLS {
            let q = (cy / 2) * 2 + cx / 2;
            for c in 0..3 {
                k[q * 3 + c] += d[(cy * CELLS + cx) * 3 + c] * 0.25;
            }
        }
    }
    k
}

pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_rect_fills_tile() {
        let r = crop_rect(2.0, 1.0, 0.5);
        assert_eq!(r, [0.25, 0.0, 0.5, 1.0]);
        let r = crop_rect(0.5, 1.0, 0.0);
        assert_eq!(r, [0.0, 0.0, 1.0, 0.5]);
        let r = crop_rect(1.0, 1.0, 0.5);
        assert_eq!(r, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn crop_descriptor_matches_direct_analysis_of_crop() {
        // Landscape image: left half black, right half white.
        let img = RgbImage::from_fn(160, 80, |x, _| if x < 80 { image::Rgb([0, 0, 0]) } else { image::Rgb([255, 255, 255]) });
        let d = compute(&img, 160, 80);
        // Square crop at the far right is pure white.
        let right = crop_descriptor(&d.grid, crop_rect(2.0, 1.0, 1.0));
        assert!(right.chunks(3).all(|c| c[0] > 99.0));
        let left = crop_descriptor(&d.grid, crop_rect(2.0, 1.0, 0.0));
        assert!(left.chunks(3).all(|c| c[0] < 1.0));
        // Centre crop: left two columns black, right two white.
        let mid = crop_descriptor(&d.grid, crop_rect(2.0, 1.0, 0.5));
        assert!(mid[0] < 1.0 && mid[3 * 3] > 99.0);
    }
}
