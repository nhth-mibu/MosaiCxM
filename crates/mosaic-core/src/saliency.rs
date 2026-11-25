//! Fast offline visual saliency for smart tile cropping beyond human faces.
//!
//! Computes a lightweight (~64 px) CIE Lab frequency-tuned + border-contrast +
//! gradient-energy saliency map to find the primary visual subject (pets,
//! landmarks, vehicles, flowers, products) and cache its normalised bounding
//! box in SQLite alongside face boxes.

use crate::color::{de2, srgb8_to_lab, Lab};
use image::RgbImage;

/// Bump when saliency extraction changes so existing libraries refresh on update.
pub const SALIENCY_VERSION: u32 = 1;

const MAX_DIM: usize = 64;
/// Minimum peak contrast (in Lab ΔE units) required to declare a distinct subject.
const MIN_PEAK_CONTRAST: f32 = 8.0;

/// Primary visual subject box `[x, y, w, h]` in normalised upright coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectBox {
    pub rect: [f32; 4],
    pub score: f32,
}

/// Detect the primary visual subject in an upright RGB image.
/// Returns `None` for nearly flat/uniform images or tiny frames.
pub fn detect(img: &RgbImage) -> Option<SubjectBox> {
    let (w, h) = img.dimensions();
    if w < 4 || h < 4 {
        return None;
    }
    let scale = MAX_DIM as f32 / w.max(h) as f32;
    let gw = ((w as f32 * scale).round() as usize).clamp(4, MAX_DIM);
    let gh = ((h as f32 * scale).round() as usize).clamp(4, MAX_DIM);

    // Box-downsample into a compact CIE Lab grid.
    let mut lab = vec![[0.0f32; 3]; gw * gh];
    for gy in 0..gh {
        let y0 = gy * h as usize / gh;
        let y1 = ((gy + 1) * h as usize / gh).max(y0 + 1).min(h as usize);
        for gx in 0..gw {
            let x0 = gx * w as usize / gw;
            let x1 = ((gx + 1) * w as usize / gw).max(x0 + 1).min(w as usize);
            let mut sum = [0.0f32; 3];
            let mut count = 0.0f32;
            for py in y0..y1 {
                for px in x0..x1 {
                    let p = img.get_pixel(px as u32, py as u32).0;
                    let c = srgb8_to_lab(p);
                    sum[0] += c[0];
                    sum[1] += c[1];
                    sum[2] += c[2];
                    count += 1.0;
                }
            }
            let inv = 1.0 / count.max(1.0);
            lab[gy * gw + gx] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
        }
    }

    // 3x3 separable-like box smooth to suppress single-pixel noise (Achanta FT).
    let mut smooth = vec![[0.0f32; 3]; gw * gh];
    for y in 0..gh {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(gh - 1);
        for x in 0..gw {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(gw - 1);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0f32;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let c = lab[ny * gw + nx];
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                    n += 1.0;
                }
            }
            smooth[y * gw + x] = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    }

    // Global mean Lab & 1-cell border mean Lab.
    let mut gsum = [0.0f32; 3];
    let mut bsum = [0.0f32; 3];
    let mut bcount = 0.0f32;
    for y in 0..gh {
        for x in 0..gw {
            let c = smooth[y * gw + x];
            gsum[0] += c[0];
            gsum[1] += c[1];
            gsum[2] += c[2];
            if x == 0 || y == 0 || x + 1 == gw || y + 1 == gh {
                bsum[0] += c[0];
                bsum[1] += c[1];
                bsum[2] += c[2];
                bcount += 1.0;
            }
        }
    }
    let inv_n = 1.0 / (gw * gh) as f32;
    let global_mean: Lab = [gsum[0] * inv_n, gsum[1] * inv_n, gsum[2] * inv_n];
    let inv_b = 1.0 / bcount.max(1.0);
    let border_mean: Lab = [bsum[0] * inv_b, bsum[1] * inv_b, bsum[2] * inv_b];

    // Combine border contrast, global contrast, and luminance gradient energy.
    let mut sal = vec![0.0f32; gw * gh];
    let mut max_raw = 0.0f32;
    for y in 0..gh {
        let ny = (y as f32 + 0.5) / gh as f32 - 0.5;
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(gh - 1);
        for x in 0..gw {
            let nx = (x as f32 + 0.5) / gw as f32 - 0.5;
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(gw - 1);
            let c = smooth[y * gw + x];
            let border_de = de2(&c, &border_mean).sqrt();
            let global_de = de2(&c, &global_mean).sqrt();
            let gx = (lab[y * gw + xp][0] - lab[y * gw + xm][0]).abs();
            let gy = (lab[yp * gw + x][0] - lab[ym * gw + x][0]).abs();
            let grad = 0.5 * (gx + gy);
            // Gentle center prior so noisy extreme corners don't dominate, while
            // still letting clearly off-centre subjects win easily.
            let r2 = nx * nx + ny * ny;
            let prior = 0.72 + 0.28 * (-r2 / 0.35).exp();
            let v = (0.45 * border_de + 0.35 * global_de + 0.20 * grad) * prior;
            sal[y * gw + x] = v;
            if v > max_raw {
                max_raw = v;
            }
        }
    }

    if max_raw < MIN_PEAK_CONTRAST {
        return None;
    }

    // Normalise saliency map and threshold foreground mass.
    let mut mean = 0.0f32;
    for v in &mut sal {
        *v /= max_raw;
        mean += *v;
    }
    mean *= inv_n;
    let var = sal.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * inv_n;
    let thresh = (mean + 0.75 * var.sqrt()).clamp(0.35, 0.75);

    let mut wsum = 0.0f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut min_x = 1.0f32;
    let mut min_y = 1.0f32;
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    let mut count = 0usize;
    for y in 0..gh {
        let fy0 = y as f32 / gh as f32;
        let fy1 = (y + 1) as f32 / gh as f32;
        let fyc = 0.5 * (fy0 + fy1);
        for x in 0..gw {
            let v = sal[y * gw + x];
            if v < thresh {
                continue;
            }
            let fx0 = x as f32 / gw as f32;
            let fx1 = (x + 1) as f32 / gw as f32;
            let fxc = 0.5 * (fx0 + fx1);
            let wt = (v - thresh + 0.05).powi(2);
            wsum += wt;
            cx += fxc * wt;
            cy += fyc * wt;
            min_x = min_x.min(fx0);
            min_y = min_y.min(fy0);
            max_x = max_x.max(fx1);
            max_y = max_y.max(fy1);
            count += 1;
        }
    }
    if count == 0 || wsum <= 1e-6 {
        return None;
    }
    cx /= wsum;
    cy /= wsum;
    // Clamp the box around the weighted centroid so scattered outliers don't
    // expand the subject to the entire frame.
    let bw = (max_x - min_x).clamp(0.12, 0.80);
    let bh = (max_y - min_y).clamp(0.12, 0.80);
    let x = (cx - 0.5 * bw).clamp(0.0, 1.0 - bw);
    let y = (cy - 0.5 * bh).clamp(0.0, 1.0 - bh);
    let score = (max_raw / 50.0).clamp(0.0, 1.0);
    Some(SubjectBox { rect: [x, y, bw, bh], score })
}

/// Crop position `t` in `[0, 1]` along the free axis that centres `subject`
/// inside the tile crop window. Mirrors [`crate::faces::crop_t`].
pub fn crop_t(photo_aspect: f32, tile_aspect: f32, subject: SubjectBox, fallback: f32) -> f32 {
    let [sx, sy, sw, sh] = subject.rect;
    let cx = sx + 0.5 * sw;
    let cy = sy + 0.5 * sh;
    if photo_aspect > tile_aspect * 1.01 {
        let w = (tile_aspect / photo_aspect).clamp(1e-4, 1.0);
        let slack = 1.0 - w;
        if slack <= 1e-4 {
            return fallback;
        }
        ((cx - 0.5 * w) / slack).clamp(0.0, 1.0)
    } else if tile_aspect > photo_aspect * 1.01 {
        let h = (photo_aspect / tile_aspect).clamp(1e-4, 1.0);
        let slack = 1.0 - h;
        if slack <= 1e-4 {
            return fallback;
        }
        ((cy - 0.5 * h) / slack).clamp(0.0, 1.0)
    } else {
        fallback
    }
}

/// Compact little-endian encoding for SQLite storage (0 bytes for `None`, 20 bytes for `Some`).
pub fn encode(s: Option<SubjectBox>) -> Vec<u8> {
    let Some(s) = s else { return Vec::new() };
    let mut out = Vec::with_capacity(20);
    for v in [s.rect[0], s.rect[1], s.rect[2], s.rect[3], s.score] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn decode(bytes: &[u8]) -> Option<SubjectBox> {
    if bytes.len() != 20 {
        return None;
    }
    let f = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let rect = [f(0), f(1), f(2), f(3)];
    let score = f(4);
    if rect.iter().all(|v| v.is_finite()) && score.is_finite() && rect[2] > 0.0 && rect[3] > 0.0 {
        Some(SubjectBox { rect, score })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_has_no_saliency_subject() {
        let flat = RgbImage::from_pixel(120, 80, image::Rgb([120, 130, 140]));
        assert!(detect(&flat).is_none());
    }

    #[test]
    fn off_center_subject_shifts_crop_to_subject() {
        // Wide 2:1 blue sky with a bright orange subject on the right side.
        let mut img = RgbImage::from_pixel(200, 100, image::Rgb([40, 100, 200]));
        for y in 25..75 {
            for x in 145..185 {
                img.put_pixel(x, y, image::Rgb([245, 130, 20]));
            }
        }
        let subj = detect(&img).expect("should detect salient orange subject");
        let cx = subj.rect[0] + 0.5 * subj.rect[2];
        assert!(cx > 0.70, "subject centroid should be on the right, got {cx}");
        let t = crop_t(2.0, 1.0, subj, 0.5);
        assert!(t > 0.70, "crop_t should shift right toward subject, got {t}");

        let enc = encode(Some(subj));
        assert_eq!(decode(&enc), Some(subj));
        assert_eq!(decode(&encode(None)), None);
    }
}
