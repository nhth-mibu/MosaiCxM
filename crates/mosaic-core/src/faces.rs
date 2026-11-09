//! Offline face detection, used to keep faces inside tile crops.
//!
//! The YuNet detector (MIT licence, see `models/README.md`) is embedded in the
//! binary and runs on the CPU with `tract`, so nothing leaves the computer.
//! Faces are found once per photo while indexing, on a small decode (long side
//! [`INPUT`] px), and stored as normalised boxes in the upright frame.

use anyhow::{anyhow, Result};
use image::RgbImage;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

/// Bump when detection changes; photos without faces of this version are
/// analysed again on the next index update.
pub const FACES_VERSION: u32 = 1;
/// Detector input (square, letterboxed). Faces smaller than about 10 px at
/// this size (3% of the long side) are not found.
pub const INPUT: usize = 320;
/// Short side (px) to decode at so that the long side is at least [`INPUT`]
/// for common photo shapes (up to 16:9).
pub const DECODE_SHORT_SIDE: u32 = 180;
const SCORE_MIN: f32 = 0.75;
const NMS_IOU: f32 = 0.3;
/// Only the largest faces are kept.
pub const MAX_FACES: usize = 8;

static MODEL_BYTES: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");

/// A face box `[x, y, w, h]` in normalised upright-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub rect: [f32; 4],
    pub score: f32,
}

type Model = std::sync::Arc<TypedRunnableModel>;

fn model() -> Result<&'static Model> {
    static M: OnceLock<std::result::Result<Model, String>> = OnceLock::new();
    M.get_or_init(|| {
        tract_onnx::onnx()
            // The file's stored shapes are for 640×640; the network itself is size-agnostic.
            .with_ignore_output_shapes(true)
            .with_ignore_value_info(true)
            .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
            .and_then(|m| m.with_input_fact(0, f32::fact([1, 3, INPUT, INPUT]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("face detector: {e}"))
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

/// Faces in an upright image, largest first (at most [`MAX_FACES`]).
pub fn detect(img: &RgbImage) -> Result<Vec<Face>> {
    let (w, h) = img.dimensions();
    if w < 2 || h < 2 {
        return Ok(Vec::new());
    }
    let s = INPUT as f32 / w.max(h) as f32;
    let (rw, rh) = (((w as f32 * s).round() as u32).clamp(1, INPUT as u32), ((h as f32 * s).round() as u32).clamp(1, INPUT as u32));
    let small = if (rw, rh) == (w, h) { img.clone() } else { crate::decode::resize_crop(img, [0.0, 0.0, 1.0, 1.0], rw, rh)? };
    // Letterboxed BGR planes, 0–255, as YuNet expects.
    let plane = INPUT * INPUT;
    let mut x = vec![0f32; 3 * plane];
    for (px, py, p) in small.enumerate_pixels() {
        let i = py as usize * INPUT + px as usize;
        x[i] = p[2] as f32;
        x[plane + i] = p[1] as f32;
        x[2 * plane + i] = p[0] as f32;
    }
    let input = Tensor::from_shape(&[1, 3, INPUT, INPUT], &x)?;
    let out = model()?.run(tvec!(input.into()))?;
    let view = |i: usize| out.get(i).ok_or_else(|| anyhow!("face detector: missing output")).and_then(|t| Ok(t.to_plain_array_view::<f32>()?.iter().copied().collect::<Vec<f32>>()));
    let mut found = Vec::new();
    // Outputs: cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32.
    for (k, stride) in [8usize, 16, 32].into_iter().enumerate() {
        let (cls, obj, bbox) = (view(k)?, view(3 + k)?, view(6 + k)?);
        decode_level(&cls, &obj, &bbox, stride, &mut found);
    }
    let mut faces = nms(found);
    // Back to normalised coordinates of the image, clipped to it.
    let (iw, ih) = (rw as f32, rh as f32);
    for f in &mut faces {
        let [x0, y0, bw, bh] = f.rect;
        let (x1, y1) = ((x0 + bw).min(iw), (y0 + bh).min(ih));
        let (x0, y0) = (x0.max(0.0), y0.max(0.0));
        f.rect = [x0 / iw, y0 / ih, (x1 - x0).max(0.0) / iw, (y1 - y0).max(0.0) / ih];
    }
    faces.retain(|f| f.rect[2] > 0.0 && f.rect[3] > 0.0);
    faces.sort_by(|a, b| (b.rect[2] * b.rect[3]).total_cmp(&(a.rect[2] * a.rect[3])));
    faces.truncate(MAX_FACES);
    Ok(faces)
}

/// Candidate boxes (input pixels) of one stride level of the YuNet head.
fn decode_level(cls: &[f32], obj: &[f32], bbox: &[f32], stride: usize, out: &mut Vec<Face>) {
    let cols = INPUT / stride;
    for i in 0..cls.len().min(obj.len()).min(bbox.len() / 4) {
        let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
        if score < SCORE_MIN {
            continue;
        }
        let (r, c) = ((i / cols) as f32, (i % cols) as f32);
        let b = &bbox[i * 4..i * 4 + 4];
        let s = stride as f32;
        let (cx, cy, w, h) = ((c + b[0]) * s, (r + b[1]) * s, b[2].exp() * s, b[3].exp() * s);
        out.push(Face { rect: [cx - w / 2.0, cy - h / 2.0, w, h], score });
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let iy = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    inter / (a[2] * a[3] + b[2] * b[3] - inter).max(1e-6)
}

fn nms(mut v: Vec<Face>) -> Vec<Face> {
    v.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    for f in v {
        if keep.iter().all(|k| iou(&k.rect, &f.rect) < NMS_IOU) {
            keep.push(f);
        }
    }
    keep
}

/// Compact storage: 4 × u16 per face (score is not stored).
pub fn encode(faces: &[Face]) -> Vec<u8> {
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes();
    faces.iter().take(MAX_FACES).flat_map(|f| f.rect.iter().flat_map(move |&v| q(v))).collect()
}

pub fn decode(bytes: &[u8]) -> Vec<Face> {
    bytes
        .chunks_exact(8)
        .map(|c| {
            let v = |i: usize| u16::from_le_bytes([c[i], c[i + 1]]) as f32 / 65535.0;
            Face { rect: [v(0), v(2), v(4), v(6)], score: 1.0 }
        })
        .collect()
}

/// Crop position `t` (as used by [`crate::descriptor::crop_rect`]) that keeps
/// faces whole, or None when there are no faces or nothing to slide.
///
/// If every face fits, the default crop is kept when it holds them all, else
/// the faces are centred in the crop. Otherwise the window that holds the most face
/// area wins.
pub fn crop_t(photo_aspect: f32, tile_aspect: f32, faces: &[Face], default_t: f32) -> Option<f32> {
    if faces.is_empty() || (photo_aspect - tile_aspect).abs() < 1e-4 {
        return None;
    }
    let horizontal = photo_aspect > tile_aspect;
    let span = |f: &Face| if horizontal { (f.rect[0], f.rect[0] + f.rect[2]) } else { (f.rect[1], f.rect[1] + f.rect[3]) };
    // Fraction of the axis the crop keeps, and how far it can slide.
    let keep = if horizontal { tile_aspect / photo_aspect } else { photo_aspect / tile_aspect };
    let free = 1.0 - keep;
    if free <= 1e-4 {
        return None;
    }
    let clamp = |s: f32| s.clamp(0.0, free);
    let lo = faces.iter().map(|f| span(f).0).fold(f32::MAX, f32::min);
    let hi = faces.iter().map(|f| span(f).1).fold(f32::MIN, f32::max);
    let start = if hi - lo <= keep {
        // All faces fit: the default crop if it already holds them, else a
        // crop with the faces in its middle.
        let d = default_t * free;
        if d <= lo && d + keep >= hi { d } else { clamp((lo + hi - keep) / 2.0) }
    } else {
        // Not all fit: try windows that start at a face or end at a face, and
        // keep the one with the most whole-face area.
        let mut best = (f32::MIN, default_t * free);
        for f in faces {
            let (a, b) = span(f);
            for s in [clamp(a), clamp(b - keep)] {
                let area: f32 = faces.iter().filter(|g| { let (ga, gb) = span(g); ga >= s - 1e-4 && gb <= s + keep + 1e-4 }).map(|g| g.rect[2] * g.rect[3]).sum();
                if area > best.0 + 1e-9 {
                    best = (area, s);
                }
            }
        }
        best.1
    };
    Some((clamp(start) / free).clamp(0.0, 1.0))
}

/// Total area of the faces that lie wholly inside a crop rectangle.
pub fn face_area_in(crop: [f32; 4], faces: &[Face]) -> f32 {
    const EPS: f32 = 1e-4;
    faces
        .iter()
        .filter(|f| {
            let [fx, fy, fw, fh] = f.rect;
            fx >= crop[0] - EPS && fy >= crop[1] - EPS && fx + fw <= crop[0] + crop[2] + EPS && fy + fh <= crop[1] + crop[3] + EPS
        })
        .map(|f| f.rect[2] * f.rect[3])
        .sum()
}

/// Whether a crop rectangle cuts through any face (holds part, not all, of it).
pub fn cuts_face(crop: [f32; 4], faces: &[Face]) -> bool {
    const EPS: f32 = 1e-3;
    faces.iter().any(|f| {
        let [fx, fy, fw, fh] = f.rect;
        let ix = ((fx + fw).min(crop[0] + crop[2]) - fx.max(crop[0])).max(0.0);
        let iy = ((fy + fh).min(crop[1] + crop[3]) - fy.max(crop[1])).max(0.0);
        let inter = ix * iy;
        inter > EPS * fw * fh && inter < (1.0 - EPS) * fw * fh
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::crop_rect;

    fn face(x: f32, y: f32, w: f32, h: f32) -> Face {
        Face { rect: [x, y, w, h], score: 1.0 }
    }

    #[test]
    fn encode_roundtrip() {
        let f = vec![face(0.1, 0.2, 0.3, 0.4), face(0.0, 0.5, 1.0, 0.5)];
        let d = decode(&encode(&f));
        assert_eq!(d.len(), 2);
        for (a, b) in f.iter().zip(&d) {
            for k in 0..4 {
                assert!((a.rect[k] - b.rect[k]).abs() < 1e-4);
            }
        }
        assert!(decode(&[]).is_empty());
    }

    #[test]
    fn crop_moves_to_keep_a_low_face() {
        // Tall 1:2 photo in a square tile; face near the bottom.
        let faces = [face(0.3, 0.7, 0.3, 0.15)];
        let t = crop_t(0.5, 1.0, &faces, 0.2).unwrap();
        let r = crop_rect(0.5, 1.0, t);
        assert!(!cuts_face(r, &faces));
        assert!(r[1] <= 0.7 && r[1] + r[3] >= 0.85, "{r:?}");
        // The default top crop would have cut it off entirely; a face already
        // inside the default crop leaves it unchanged.
        let top = [face(0.3, 0.1, 0.3, 0.15)];
        assert_eq!(crop_t(0.5, 1.0, &top, 0.2), Some(0.2));
    }

    #[test]
    fn crop_slides_sideways_for_wide_photos() {
        // 2:1 landscape, square tile, person on the far right.
        let faces = [face(0.8, 0.3, 0.1, 0.2)];
        let t = crop_t(2.0, 1.0, &faces, 0.5).unwrap();
        let r = crop_rect(2.0, 1.0, t);
        assert!(r[0] <= 0.8 && r[0] + r[2] >= 0.9, "{r:?}");
    }

    #[test]
    fn group_too_wide_keeps_most_face_area() {
        // 3:1 landscape: crop keeps one third. Big face left, small face right.
        let faces = [face(0.05, 0.3, 0.2, 0.4), face(0.85, 0.3, 0.05, 0.1)];
        let t = crop_t(3.0, 1.0, &faces, 0.5).unwrap();
        let r = crop_rect(3.0, 1.0, t);
        assert!(r[0] <= 0.05 && r[0] + r[2] >= 0.25, "{r:?}");
    }

    #[test]
    fn no_faces_or_same_shape_means_no_change() {
        assert_eq!(crop_t(0.5, 1.0, &[], 0.2), None);
        assert_eq!(crop_t(1.0, 1.0, &[face(0.0, 0.0, 0.5, 0.5)], 0.5), None);
    }

    #[test]
    fn cut_detection() {
        let f = [face(0.4, 0.4, 0.2, 0.2)];
        assert!(!cuts_face([0.0, 0.0, 1.0, 1.0], &f));
        assert!(!cuts_face([0.0, 0.0, 0.3, 1.0], &f));
        assert!(cuts_face([0.0, 0.0, 0.5, 1.0], &f));
    }

    #[test]
    fn detector_runs_and_finds_nothing_in_a_flat_image() {
        let img = RgbImage::from_pixel(300, 200, image::Rgb([120, 140, 160]));
        assert!(detect(&img).unwrap().is_empty());
    }

    /// Run with `MOSAIC_FACE_SAMPLE=/path/to/portrait.jpg cargo test -p mosaic-core -- --ignored`.
    #[test]
    #[ignore]
    fn detector_finds_a_face_in_a_sample() {
        let p = std::env::var("MOSAIC_FACE_SAMPLE").expect("set MOSAIC_FACE_SAMPLE");
        let d = crate::decode::decode_file(std::path::Path::new(&p), Some(DECODE_SHORT_SIDE)).unwrap();
        let f = detect(&d.image).unwrap();
        eprintln!("{f:?}");
        assert!(!f.is_empty());
    }
}
