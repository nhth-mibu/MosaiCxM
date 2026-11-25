//! Candidate retrieval and tile assignment for the three mosaic methods.
//!
//! Pipeline (shared by all methods):
//! 1. Build candidates: every pool photo × allowed crop positions, each with a
//!    4×4 Lab descriptor derived from the stored grid (no decoding).
//! 2. Retrieve a small candidate set per tile from a k-d tree (mean Lab for
//!    Classic, 2×2 Lab for the others) instead of scanning the whole pool.
//! 3. Rerank those candidates with the method's score.
//! 4. Assign tiles one by one in a seeded random order, honouring hard reuse
//!    limits and adding method-specific penalties. If all retrieved candidates
//!    of a tile are used up, the search widens (×8) and finally falls back to
//!    a full scan, so a narrow search never causes a false "infeasible".

use crate::color::{de2, Lab};
use crate::descriptor::{self, crop_descriptor, crop_positions, crop_rect, key12, mean_lab, TileDesc, CELLS};
use crate::job::Job;
use crate::faces;
use crate::pool::{Pool, PoolPhoto};
use crate::rng::SplitMix64;
use anyhow::{bail, Result};
use image::RgbImage;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::num::NonZero;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Classic,
    Structure,
    Variety,
}

impl Method {
    pub const ALL: [Method; 3] = [Method::Classic, Method::Structure, Method::Variety];
    pub fn name(self) -> &'static str {
        match self {
            Method::Classic => "classic",
            Method::Structure => "structure",
            Method::Variety => "variety",
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Method::Classic => "Classic colour mosaic",
            Method::Structure => "Structure-aware mosaic",
            Method::Variety => "Variety-aware mosaic",
        }
    }
    pub fn explanation(self) -> &'static str {
        match self {
            Method::Classic => "Picks the photo whose average colour is closest to each tile of the motif. Fastest; reproduces large areas of colour well but ignores what is inside each tile.",
            Method::Structure => "Compares a 4×4 colour layout and light/dark edges inside every tile, so photos with a matching edge or gradient are preferred. Better at outlines and local detail.",
            Method::Variety => "Like structure-aware, but every reuse of a photo costs extra and repeats near each other are penalised, so more different photos appear. Slightly less exact colour.",
        }
    }
    pub fn parse(s: &str) -> Result<Method> {
        match s.to_ascii_lowercase().as_str() {
            "classic" | "a" | "color" | "colour" => Ok(Method::Classic),
            "structure" | "b" => Ok(Method::Structure),
            "variety" | "c" => Ok(Method::Variety),
            _ => bail!("unknown method '{s}' (classic, structure, variety)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchSettings {
    pub method: Method,
    /// Hard limit on uses per photo (duplicate group). None = unlimited.
    pub max_reuse: Option<u32>,
    /// Extra cost (ΔE) for using the same photo again within `repeat_radius` tiles.
    pub repeat_penalty: f32,
    pub repeat_radius: u32,
    /// Extra cost (ΔE) per previous use of a photo (variety method).
    pub usage_penalty: f32,
    /// Also consider off-centre crops (start/end of the long side).
    pub crop_offsets: bool,
    /// Place crops so faces found while indexing stay whole.
    #[serde(default = "default_true")]
    pub face_crop: bool,
    pub merge_near_duplicates: bool,
    pub seed: u64,
    /// Source-photo restrictions, saved with the project by the desktop app.
    #[serde(default)]
    pub photo_filters: crate::filters::PhotoFilters,
    /// Saved mosaics made with earlier document checks need an explicit refresh.
    #[serde(default)]
    pub document_filter_version: u32,
    /// Candidates retrieved per tile before reranking.
    pub candidates: usize,
    /// 0 = off. Fraction (≤ 0.5) of each tile's mean-colour error corrected at render time.
    pub color_correction: f32,
    /// 0 = off. Opacity (≤ 0.5) of the motif blended over the mosaic at render time.
    pub motif_blend: f32,
    /// Text written into the mosaic with tile brightness (see [`crate::message`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_message: Option<crate::message::HiddenMessage>,
    /// Content hashes (64-char hex) of hero photos guaranteed to appear at least once.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must_include: Vec<String>,
    /// Motif crop zoom factor (1.0 = full cover fit, up to 5.0).
    #[serde(default = "default_motif_zoom", skip_serializing_if = "is_default_motif_zoom")]
    pub motif_zoom: f32,
    /// Motif crop pan position `[x, y]` in `0.0..=1.0` (`[0.5, 0.5]` = centred).
    #[serde(default = "default_motif_pan", skip_serializing_if = "is_default_motif_pan")]
    pub motif_pan: [f32; 2],
    /// Assign tiles covering motif faces and high-detail edges before flat regions.
    #[serde(default, skip_serializing_if = "is_false")]
    pub focal_priority: bool,
    /// Content hashes (64-char hex) of photos excluded from this mosaic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_photos: Vec<String>,
    /// Row-major indices of tiles locked across photo reselection and swaps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pinned_tiles: Vec<u32>,
}

fn default_motif_zoom() -> f32 {
    1.0
}
fn is_default_motif_zoom(z: &f32) -> bool {
    (*z - 1.0).abs() < 1e-4
}
fn default_motif_pan() -> [f32; 2] {
    [0.5, 0.5]
}
fn is_default_motif_pan(p: &[f32; 2]) -> bool {
    (p[0] - 0.5).abs() < 1e-4 && (p[1] - 0.5).abs() < 1e-4
}
fn is_false(b: &bool) -> bool {
    !*b
}

impl MatchSettings {
    pub fn for_method(method: Method) -> Self {
        let mut s = Self {
            method,
            max_reuse: None,
            repeat_penalty: 0.0,
            repeat_radius: 2,
            usage_penalty: 0.0,
            crop_offsets: false,
            face_crop: true,
            merge_near_duplicates: true,
            seed: 1,
            photo_filters: crate::filters::PhotoFilters::default(),
            document_filter_version: crate::filters::DOCUMENT_VERSION,
            candidates: 48,
            color_correction: 0.0,
            motif_blend: 0.0,
            hidden_message: None,
            must_include: Vec::new(),
            motif_zoom: 1.0,
            motif_pan: [0.5, 0.5],
            focal_priority: false,
            excluded_photos: Vec::new(),
            pinned_tiles: Vec::new(),
        };
        if method == Method::Variety {
            s.repeat_penalty = 8.0;
            s.repeat_radius = 3;
            s.usage_penalty = 1.5;
            s.crop_offsets = true;
            s.candidates = 96;
        }
        s
    }
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=0.5).contains(&self.color_correction) {
            bail!("colour correction must be between 0 and 0.5");
        }
        if !(0.0..=0.5).contains(&self.motif_blend) {
            bail!("motif blend must be between 0 and 0.5");
        }
        if self.max_reuse == Some(0) {
            bail!("maximum reuse must be at least 1 (or unlimited)");
        }
        if self.repeat_penalty < 0.0 || self.usage_penalty < 0.0 {
            bail!("penalties must not be negative");
        }
        if self.repeat_radius > 20 {
            bail!("repeat radius must be at most 20 tiles");
        }
        if !self.motif_zoom.is_finite() || !(1.0..=5.0).contains(&self.motif_zoom) {
            bail!("motif zoom must be between 1 and 5");
        }
        if !self.motif_pan.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            bail!("motif pan must be between 0 and 1");
        }
        self.photo_filters.validate()?;
        for h in &self.must_include {
            crate::catalog::unhex(h).map_err(|_| anyhow::anyhow!("invalid must-include photo hash '{h}'"))?;
        }
        for h in &self.excluded_photos {
            crate::catalog::unhex(h).map_err(|_| anyhow::anyhow!("invalid excluded photo hash '{h}'"))?;
        }
        if let Some(m) = &self.hidden_message {
            m.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridSpec {
    pub cols: u32,
    pub rows: u32,
    pub width_mm: f64,
    pub height_mm: f64,
}

impl GridSpec {
    pub fn tiles(&self) -> usize {
        self.cols as usize * self.rows as usize
    }
    pub fn tile_aspect(&self) -> f32 {
        ((self.width_mm / self.cols as f64) / (self.height_mm / self.rows as f64)) as f32
    }
    pub fn tile_mm(&self) -> (f64, f64) {
        (self.width_mm / self.cols as f64, self.height_mm / self.rows as f64)
    }
    /// Grid from physical size and an approximate tile size (square-ish tiles).
    pub fn from_tile_mm(width_mm: f64, height_mm: f64, tile_mm: f64) -> Result<Self> {
        crate::units::validate_artwork_mm(width_mm, height_mm)?;
        if !(tile_mm > 0.0) {
            bail!("tile size must be positive");
        }
        let cols = (width_mm / tile_mm).round().max(1.0) as u32;
        let rows = (height_mm / tile_mm).round().max(1.0) as u32;
        Self::new(width_mm, height_mm, cols, rows)
    }
    pub fn new(width_mm: f64, height_mm: f64, cols: u32, rows: u32) -> Result<Self> {
        crate::units::validate_artwork_mm(width_mm, height_mm)?;
        if cols == 0 || rows == 0 {
            bail!("grid must have at least one column and row");
        }
        if cols as u64 * rows as u64 > 2_000_000 {
            bail!("{cols}×{rows} tiles is above the 2,000,000 tile limit");
        }
        Ok(Self { cols, rows, width_mm, height_mm })
    }
}

/// Normalised motif crop rectangle `[x, y, w, h]` for a cover-fit zoomed by
/// `zoom` (1..=5) and panned by `pan` (`[0..=1, 0..=1]`).
pub fn motif_crop_rect(motif_aspect: f32, mosaic_aspect: f32, zoom: f32, pan: [f32; 2]) -> [f32; 4] {
    let base = crop_rect(motif_aspect, mosaic_aspect, 0.5);
    let z = if zoom.is_finite() { zoom.clamp(1.0, 5.0) } else { 1.0 };
    let w = (base[2] / z).clamp(1e-3, 1.0);
    let h = (base[3] / z).clamp(1e-3, 1.0);
    let px = if pan[0].is_finite() { pan[0].clamp(0.0, 1.0) } else { 0.5 };
    let py = if pan[1].is_finite() { pan[1].clamp(0.0, 1.0) } else { 0.5 };
    let x = ((1.0 - w) * px).clamp(0.0, 1.0 - w);
    let y = ((1.0 - h) * py).clamp(0.0, 1.0 - h);
    [x, y, w, h]
}

/// Target motif sampled at CELLS×CELLS Lab cells per tile.
#[derive(Debug, Clone)]
pub struct Target {
    pub cols: u32,
    pub rows: u32,
    pub tiles: Vec<TileDesc>,
    /// Per-tile visual priority (motif faces + edge/contrast salience), used
    /// when [`MatchSettings::focal_priority`] is enabled.
    pub tile_priority: Vec<f32>,
}

/// Compute per-tile visual salience from motif contrast plus optional face
/// boxes (in normalised cropped-motif coordinates). Higher values match first.
pub fn tile_salience(target: &Target, faces: &[crate::faces::Face]) -> Vec<f32> {
    let cols = target.cols as usize;
    let rows = target.rows as usize;
    let n = cols * rows;
    if n == 0 || target.tiles.len() != n {
        return Vec::new();
    }
    let means: Vec<Lab> = target.tiles.iter().map(mean_lab).collect();
    let mut raw = vec![0.0f32; n];
    let mut max_raw = 0.0f32;
    let cell_count = (CELLS * CELLS) as f32;
    for y in 0..rows {
        for x in 0..cols {
            let idx = y * cols + x;
            let d = &target.tiles[idx];
            let m = means[idx];
            // Internal luminance + chroma detail within the tile.
            let mut internal = 0.0f32;
            for c in 0..(CELLS * CELLS) {
                let dl = d[c * 3] - m[0];
                let da = d[c * 3 + 1] - m[1];
                let db = d[c * 3 + 2] - m[2];
                internal += (dl * dl + 0.5 * (da * da + db * db)).sqrt();
            }
            internal /= cell_count;
            // Edge contrast against 4-neighbours.
            let mut edge = 0.0f32;
            let mut nc = 0.0f32;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < cols && ny < rows {
                    edge += de2(&m, &means[ny * cols + nx]).sqrt();
                    nc += 1.0;
                }
            }
            if nc > 0.0 {
                edge /= nc;
            }
            let v = internal + 0.5 * edge;
            raw[idx] = v;
            if v > max_raw {
                max_raw = v;
            }
        }
    }
    let scale = if max_raw > 1e-4 { 1.0 / max_raw } else { 0.0 };
    let mut prio: Vec<f32> = raw.into_iter().map(|v| v * scale).collect();
    // Strongly boost tiles overlapping any detected face in the motif.
    if !faces.is_empty() {
        let fw = 1.0 / cols as f32;
        let fh = 1.0 / rows as f32;
        for y in 0..rows {
            let ty0 = y as f32 * fh;
            let ty1 = ty0 + fh;
            for x in 0..cols {
                let tx0 = x as f32 * fw;
                let tx1 = tx0 + fw;
                let hits_face = faces.iter().any(|f| {
                    let [fx, fy, fbw, fbh] = f.rect;
                    tx0 < fx + fbw && tx1 > fx && ty0 < fy + fbh && ty1 > fy
                });
                if hits_face {
                    prio[y * cols + x] += 2.0;
                }
            }
        }
    }
    prio
}

impl Target {
    /// `crop` is the normalised part of the motif used (centred cover fit).
    pub fn from_image(img: &RgbImage, grid: &GridSpec) -> (Self, [f32; 4]) {
        Self::from_image_cropped(img, grid, 1.0, [0.5, 0.5])
    }

    /// Like [`Target::from_image`] with a custom `zoom` (1..=5) and `pan` (`[0..=1, 0..=1]`).
    pub fn from_image_cropped(img: &RgbImage, grid: &GridSpec, zoom: f32, pan: [f32; 2]) -> (Self, [f32; 4]) {
        let motif_aspect = img.width() as f32 / img.height().max(1) as f32;
        let mosaic_aspect = (grid.width_mm / grid.height_mm) as f32;
        let crop = motif_crop_rect(motif_aspect, mosaic_aspect, zoom, pan);
        let x = ((crop[0] * img.width() as f32).round() as u32).min(img.width().saturating_sub(1));
        let y = ((crop[1] * img.height() as f32).round() as u32).min(img.height().saturating_sub(1));
        let w = ((crop[2] * img.width() as f32).round() as u32).clamp(1, img.width() - x);
        let h = ((crop[3] * img.height() as f32).round() as u32).clamp(1, img.height() - y);
        let sub = image::imageops::crop_imm(img, x, y, w, h).to_image();
        let gw = grid.cols as usize * CELLS;
        let gh = grid.rows as usize * CELLS;
        let cells = descriptor::area_average_lab(&sub, gw, gh);
        let mut target = Self::from_cells(grid.cols, grid.rows, &cells);
        let faces = crate::faces::detect(&sub).unwrap_or_default();
        target.tile_priority = tile_salience(&target, &faces);
        (target, crop)
    }

    pub fn from_cells(cols: u32, rows: u32, cells: &[Lab]) -> Self {
        let gw = cols as usize * CELLS;
        let mut tiles = Vec::with_capacity(cols as usize * rows as usize);
        for ty in 0..rows as usize {
            for tx in 0..cols as usize {
                let mut d = [0f32; descriptor::DESC_DIM];
                for cy in 0..CELLS {
                    for cx in 0..CELLS {
                        let c = cells[(ty * CELLS + cy) * gw + tx * CELLS + cx];
                        d[(cy * CELLS + cx) * 3..(cy * CELLS + cx) * 3 + 3].copy_from_slice(&c);
                    }
                }
                tiles.push(d);
            }
        }
        Self { cols, rows, tiles, tile_priority: Vec::new() }
    }

    pub fn cells(&self) -> Vec<Lab> {
        let gw = self.cols as usize * CELLS;
        let mut out = vec![[0f32; 3]; gw * self.rows as usize * CELLS];
        for (i, d) in self.tiles.iter().enumerate() {
            let (tx, ty) = (i % self.cols as usize, i / self.cols as usize);
            for cy in 0..CELLS {
                for cx in 0..CELLS {
                    let o = (cy * CELLS + cx) * 3;
                    out[(ty * CELLS + cy) * gw + tx * CELLS + cx] = [d[o], d[o + 1], d[o + 2]];
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub photo: u32,
    pub group: u32,
    pub crop: [f32; 4],
    pub desc: TileDesc,
    pub mean: Lab,
}

fn default_true() -> bool {
    true
}

/// Crop positions for one photo. With `face_crop`, the first position keeps
/// its faces whole and other positions are kept only if they cut no face and
/// hold as much face as the first.
pub fn photo_crop_positions(p: &PoolPhoto, tile_aspect: f32, crop_offsets: bool, face_crop: bool) -> Vec<f32> {
    let base = crop_positions(p.aspect(), tile_aspect, crop_offsets);
    if !face_crop || p.faces.is_empty() {
        return base.to_vec();
    }
    let Some(t) = faces::crop_t(p.aspect(), tile_aspect, &p.faces, base[0]) else { return base.to_vec() };
    let held = faces::face_area_in(crop_rect(p.aspect(), tile_aspect, t), &p.faces);
    let mut v = vec![t];
    for &o in base {
        let r = crop_rect(p.aspect(), tile_aspect, o);
        if (o - t).abs() > 1e-3 && !faces::cuts_face(r, &p.faces) && faces::face_area_in(r, &p.faces) >= held - 1e-6 {
            v.push(o);
        }
    }
    v
}

pub fn build_candidates(pool: &Pool, tile_aspect: f32, crop_offsets: bool, face_crop: bool) -> Vec<Candidate> {
    pool.photos
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, p)| {
            photo_crop_positions(p, tile_aspect, crop_offsets, face_crop).into_iter().map(move |t| {
                let crop = crop_rect(p.aspect(), tile_aspect, t);
                let desc = crop_descriptor(&p.grid, crop);
                Candidate { photo: i as u32, group: p.group, crop, mean: mean_lab(&desc), desc }
            })
        })
        .collect()
}

/// Nearest-neighbour retrieval over candidate keys.
pub enum Retrieval {
    Mean(ImmutableKdTree<f32, 3>),
    Layout(ImmutableKdTree<f32, 12>),
    /// Exhaustive scan baseline (benchmarks / tiny pools).
    Brute { layout: bool },
}

impl Retrieval {
    pub fn build(cands: &[Candidate], method: Method) -> Result<Self> {
        Ok(match method {
            Method::Classic => {
                let pts: Vec<[f32; 3]> = cands.iter().map(|c| c.mean).collect();
                Retrieval::Mean(ImmutableKdTree::new_from_slice(&pts).map_err(|e| anyhow::anyhow!("kd-tree: {e:?}"))?)
            }
            _ => {
                let pts: Vec<[f32; 12]> = cands.iter().map(|c| key12(&c.desc)).collect();
                Retrieval::Layout(ImmutableKdTree::new_from_slice(&pts).map_err(|e| anyhow::anyhow!("kd-tree: {e:?}"))?)
            }
        })
    }

    pub fn query(&self, cands: &[Candidate], target: &TileDesc, k: usize) -> Vec<u32> {
        self.query_k(cands, target, k.clamp(1, cands.len().max(1)))
    }

    fn query_k(&self, cands: &[Candidate], target: &TileDesc, k: usize) -> Vec<u32> {
        let k = k.max(1);
        let nz = NonZero::new(k).unwrap();
        match self {
            Retrieval::Mean(t) => t.query(&mean_lab(target)).nearest_n::<SquaredEuclidean<f32>>(nz).execute().into_iter().map(|r| r.item).collect(),
            Retrieval::Layout(t) => t.query(&key12(target)).nearest_n::<SquaredEuclidean<f32>>(nz).execute().into_iter().map(|r| r.item).collect(),
            Retrieval::Brute { layout } => {
                let mut v: Vec<(f32, u32)> = if *layout {
                    let tk = key12(target);
                    cands.iter().enumerate().map(|(i, c)| (dist12(&tk, &key12(&c.desc)), i as u32)).collect()
                } else {
                    let tm = mean_lab(target);
                    cands.iter().enumerate().map(|(i, c)| (de2(&tm, &c.mean), i as u32)).collect()
                };
                if k < v.len() {
                    v.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    v.truncate(k);
                }
                v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                v.into_iter().map(|x| x.1).collect()
            }
        }
    }
}

fn dist12(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Method score in ΔE units (lower is better).
pub fn score(method: Method, target: &TileDesc, cand: &Candidate) -> f32 {
    match method {
        Method::Classic => de2(&mean_lab(target), &cand.mean).sqrt(),
        Method::Structure | Method::Variety => structure_cost(target, &cand.desc),
    }
}

/// RMS cell colour error plus a term for mismatched light/dark edges.
pub fn structure_cost(t: &TileDesc, c: &TileDesc) -> f32 {
    let mut cell = 0f32;
    for i in 0..CELLS * CELLS {
        cell += de2(&[t[i * 3], t[i * 3 + 1], t[i * 3 + 2]], &[c[i * 3], c[i * 3 + 1], c[i * 3 + 2]]);
    }
    cell /= (CELLS * CELLS) as f32;
    let l = |d: &TileDesc, x: usize, y: usize| d[(y * CELLS + x) * 3];
    let mut grad = 0f32;
    let mut n = 0f32;
    for y in 0..CELLS {
        for x in 0..CELLS {
            if x + 1 < CELLS {
                let g = (l(t, x + 1, y) - l(t, x, y)) - (l(c, x + 1, y) - l(c, x, y));
                grad += g * g;
                n += 1.0;
            }
            if y + 1 < CELLS {
                let g = (l(t, x, y + 1) - l(t, x, y)) - (l(c, x, y + 1) - l(c, x, y));
                grad += g * g;
                n += 1.0;
            }
        }
    }
    (cell + 0.5 * grad / n).sqrt()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TileAssign {
    /// Index into the pool (or into `Project::sources` once saved).
    #[serde(rename = "p")]
    pub photo: u32,
    /// Normalised crop of the upright photo: x, y, w, h.
    #[serde(rename = "c")]
    pub crop: [f32; 4],
    /// Lab offset applied at render time (all zero when colour correction is off).
    #[serde(rename = "a", default, skip_serializing_if = "is_zero3")]
    pub adjust: [f32; 3],
}

impl TileAssign {
    /// Tile outside the mosaic outline: nothing is placed, background is shown.
    pub const EMPTY: TileAssign = TileAssign { photo: u32::MAX, crop: [0.0; 4], adjust: [0.0; 3] };
    pub fn is_empty(&self) -> bool {
        self.photo == u32::MAX
    }
}

fn is_zero3(a: &[f32; 3]) -> bool {
    a.iter().all(|v| *v == 0.0)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchStats {
    pub pool_photos: usize,
    pub pool_groups: u32,
    pub candidates: usize,
    pub tiles: usize,
    pub build_ms: f64,
    pub retrieval_ms: f64,
    pub assign_ms: f64,
    pub widened_searches: usize,
    /// Rebuilds of the "photos with remaining capacity" index after widening.
    #[serde(alias = "full_scans")]
    pub index_rebuilds: usize,
    pub distinct_used: usize,
    pub max_uses: u32,
    /// Mean colour error (ΔE76, tile mean vs photo crop mean) before correction.
    pub mean_delta_e: f32,
    /// Mean method cost.
    pub mean_cost: f32,
    pub nearby_repeats: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Infeasible {
    pub tiles: usize,
    pub distinct_photos: u32,
    pub max_reuse: u32,
    pub message: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum MatchError {
    #[error("{}", .0.message)]
    Infeasible(Infeasible),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub fn check_feasible(grid: &GridSpec, groups: u32, s: &MatchSettings) -> Result<(), Infeasible> {
    check_feasible_n(grid, grid.tiles(), groups, s)
}

/// Like [`check_feasible`] for `tiles` placed tiles (fewer when an outline is used).
pub fn check_feasible_n(grid: &GridSpec, tiles: usize, groups: u32, s: &MatchSettings) -> Result<(), Infeasible> {
    let Some(m) = s.max_reuse else { return Ok(()) };
    let capacity = groups as u64 * m as u64;
    if capacity >= tiles as u64 {
        return Ok(());
    }
    let needed = (tiles as u64).div_ceil(groups.max(1) as u64);
    let aspect = grid.width_mm / grid.height_mm;
    let rows = ((capacity as f64 / aspect).sqrt().floor() as u64).max(1);
    let cols = ((rows as f64 * aspect).floor() as u64).max(1);
    let (cols, rows) = if cols * rows > capacity { (cols.saturating_sub(1).max(1), rows) } else { (cols, rows) };
    let more = (tiles as u64).div_ceil(m as u64).saturating_sub(groups as u64);
    Err(Infeasible {
        tiles,
        distinct_photos: groups,
        max_reuse: m,
        message: format!(
            "Not enough distinct photos: {tiles} tiles need at least {} distinct photos at max {m} use(s) each, but the selected libraries have {groups}{}.",
            (tiles as u64).div_ceil(m as u64),
            if s.merge_near_duplicates { " after merging duplicates" } else { "" }
        ),
        suggestions: vec![
            format!("Raise the maximum reuse to {needed} or more (or unlimited)."),
            format!("Use a coarser grid with at most {capacity} tiles, e.g. {cols}×{rows}."),
            format!("Add at least {more} more distinct photos to the selected libraries."),
        ],
    })
}

/// Match a target against a pool. Deterministic for a given seed.
pub fn match_tiles(pool: &Pool, target: &Target, grid: &GridSpec, s: &MatchSettings, job: &Job) -> Result<(Vec<TileAssign>, MatchStats), MatchError> {
    match_tiles_masked(pool, target, grid, s, None, job)
}

/// Match only the `active` tiles (row-major; None = all). Inactive tiles
/// become [`TileAssign::EMPTY`].
pub fn match_tiles_masked(
    pool: &Pool,
    target: &Target,
    grid: &GridSpec,
    s: &MatchSettings,
    active: Option<&[bool]>,
    job: &Job,
) -> Result<(Vec<TileAssign>, MatchStats), MatchError> {
    match_selection(pool, target, grid, s, active, None, job)
}

/// Choose alternatives to existing pool-indexed assignments. `only` limits
/// the change to one tile; all other placements remain exactly as supplied.
/// Pinned tiles (in [`MatchSettings::pinned_tiles`]) are kept unchanged.
pub fn reselect_tiles(
    pool: &Pool,
    target: &Target,
    grid: &GridSpec,
    s: &MatchSettings,
    previous: &[TileAssign],
    only: Option<usize>,
    job: &Job,
) -> Result<(Vec<TileAssign>, MatchStats), MatchError> {
    if previous.len() != grid.tiles() || only.is_some_and(|i| previous.get(i).is_none_or(TileAssign::is_empty)) {
        return Err(anyhow::anyhow!("choose a photo tile inside the mosaic").into());
    }
    let (replace, allow_keep_old) = match only {
        Some(idx) => {
            if s.pinned_tiles.contains(&(idx as u32)) {
                return Err(anyhow::anyhow!("this tile is pinned; unpin it before swapping its photo").into());
            }
            ((0..grid.tiles()).map(|i| i == idx).collect::<Vec<_>>(), false)
        }
        None => {
            let pinned: std::collections::HashSet<u32> = s.pinned_tiles.iter().copied().collect();
            let excluded: std::collections::HashSet<crate::catalog::Hash> = s
                .excluded_photos
                .iter()
                .filter_map(|h| crate::catalog::unhex(h).ok())
                .collect();
            let mask: Vec<bool> = previous
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if t.is_empty() {
                        return false;
                    }
                    let is_excluded = pool.photos.get(t.photo as usize).is_some_and(|p| excluded.contains(&p.hash));
                    is_excluded || !pinned.contains(&(i as u32))
                })
                .collect();
            if !mask.iter().any(|&r| r) {
                return Err(anyhow::anyhow!("all tiles are pinned; unpin at least one tile to choose new photos").into());
            }
            (mask, true)
        }
    };
    reselect_tiles_subset(pool, target, grid, s, previous, &replace, allow_keep_old, job)
}

/// Replace only the tiles where `replace[i]` is true, keeping all other
/// non-empty tiles from `previous` locked in place.
#[allow(clippy::too_many_arguments)]
pub fn reselect_tiles_subset(
    pool: &Pool,
    target: &Target,
    grid: &GridSpec,
    s: &MatchSettings,
    previous: &[TileAssign],
    replace: &[bool],
    allow_keep_old: bool,
    job: &Job,
) -> Result<(Vec<TileAssign>, MatchStats), MatchError> {
    if previous.len() != grid.tiles() || replace.len() != grid.tiles() {
        return Err(anyhow::anyhow!("choose a photo tile inside the mosaic").into());
    }
    let active: Vec<bool> = previous.iter().map(|t| !t.is_empty()).collect();
    match_selection(pool, target, grid, s, Some(&active), Some((previous, replace, allow_keep_old)), job)
}

#[allow(clippy::too_many_arguments)]
fn match_selection(
    pool: &Pool,
    target: &Target,
    grid: &GridSpec,
    s: &MatchSettings,
    active: Option<&[bool]>,
    previous: Option<(&[TileAssign], &[bool], bool)>,
    job: &Job,
) -> Result<(Vec<TileAssign>, MatchStats), MatchError> {
    s.validate()?;
    if pool.photos.is_empty() {
        return Err(anyhow::anyhow!("no source photos available").into());
    }
    let all = vec![true; grid.tiles()];
    let active = active.unwrap_or(&all);
    if active.len() != grid.tiles() {
        return Err(anyhow::anyhow!("outline does not match the grid").into());
    }
    let placed_tiles = active.iter().filter(|&&a| a).count();
    if placed_tiles == 0 {
        return Err(anyhow::anyhow!("the outline leaves no tile to fill").into());
    }
    let mut excluded_group = vec![false; pool.group_count as usize];
    for hex_hash in &s.excluded_photos {
        if let Ok(h) = crate::catalog::unhex(hex_hash) {
            if let Some(pi) = pool.index_of(&h) {
                excluded_group[pool.photos[pi].group as usize] = true;
            }
        }
    }
    let excluded_count = excluded_group.iter().filter(|&&e| e).count() as u32;
    let avail_groups = pool.group_count.saturating_sub(excluded_count);
    check_feasible_n(grid, placed_tiles, avail_groups, s).map_err(MatchError::Infeasible)?;
    let mut stats = MatchStats { pool_photos: pool.photos.len(), pool_groups: avail_groups, tiles: placed_tiles, ..Default::default() };
    let t0 = Instant::now();
    job.report("building candidates", 0, 1);
    let mut cands = build_candidates(pool, grid.tile_aspect(), s.crop_offsets, s.face_crop);
    if excluded_count > 0 {
        cands.retain(|c| !excluded_group[c.group as usize]);
    }
    if cands.is_empty() {
        return Err(anyhow::anyhow!("no source photos remain after exclusions").into());
    }
    // Keep the exact saved crops available, including projects made before a
    // library's face metadata changed. Candidate IDs let assignment reuse its
    // normal capacity, neighbour and statistics calculations.
    let mut old = vec![u32::MAX; grid.tiles()];
    if let Some((tiles, _, _)) = previous {
        let key = |photo, crop: [f32; 4]| (photo, crop.map(f32::to_bits));
        let mut ids: std::collections::HashMap<_, _> = cands.iter().enumerate().map(|(i, c)| (key(c.photo, c.crop), i as u32)).collect();
        for (i, tile) in tiles.iter().enumerate().filter(|(_, t)| !t.is_empty()) {
            let photo = pool.photos.get(tile.photo as usize).ok_or_else(|| anyhow::anyhow!("a source photo is no longer indexed"))?;
            old[i] = *ids.entry(key(tile.photo, tile.crop)).or_insert_with(|| {
                let desc = crop_descriptor(&photo.grid, tile.crop);
                let id = cands.len() as u32;
                cands.push(Candidate { photo: tile.photo, group: photo.group, crop: tile.crop, mean: mean_lab(&desc), desc });
                id
            });
        }
    }
    let retrieval = Retrieval::build(&cands, s.method)?;
    stats.candidates = cands.len();
    stats.build_ms = ms(t0);
    job.check()?;
    let selected: Vec<bool> = active.iter().enumerate().map(|(i, &a)| a && previous.is_none_or(|(_, replace, _)| replace[i])).collect();
    let lists = retrieve_selection(&cands, &retrieval, target, s, Some(&selected), Some(&old), &excluded_group, job)?;
    stats.retrieval_ms = ms(t0) - stats.build_ms;
    let t1 = Instant::now();
    let tiles = assign(pool, &cands, &retrieval, target, grid, s, &selected, &lists, &old, &excluded_group, previous, &mut stats, job)?;
    stats.assign_ms = ms(t1);
    Ok((tiles, stats))
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Per tile: retrieved candidates reranked by method score (ascending).
pub fn retrieve(cands: &[Candidate], r: &Retrieval, target: &Target, s: &MatchSettings, job: &Job) -> Result<Vec<Vec<(u32, f32)>>> {
    retrieve_active(cands, r, target, s, None, job)
}

/// [`retrieve`] that leaves inactive tiles with an empty list.
pub fn retrieve_active(cands: &[Candidate], r: &Retrieval, target: &Target, s: &MatchSettings, active: Option<&[bool]>, job: &Job) -> Result<Vec<Vec<(u32, f32)>>> {
    retrieve_selection(cands, r, target, s, active, None, &[], job)
}

#[allow(clippy::too_many_arguments)]
fn retrieve_selection(cands: &[Candidate], r: &Retrieval, target: &Target, s: &MatchSettings, active: Option<&[bool]>, old: Option<&[u32]>, excluded_group: &[bool], job: &Job) -> Result<Vec<Vec<(u32, f32)>>> {
    // When a tile's choice cannot depend on earlier choices (no reuse limit,
    // no penalties), only its best candidate is ever used. Keeping just that
    // one saves about 0.4 GB per million tiles.
    let fixed = s.max_reuse.is_none() && (s.repeat_penalty <= 0.0 || s.repeat_radius == 0) && (s.method != Method::Variety || s.usage_penalty <= 0.0);
    let keep = if fixed { 1 } else { usize::MAX };
    let k = s.candidates.max(4);
    let n = target.tiles.len();
    let mut out = Vec::with_capacity(n);
    for (ci, chunk) in target.tiles.chunks(4096).enumerate() {
        job.check()?;
        let part: Vec<Vec<(u32, f32)>> = chunk
            .par_iter()
            .enumerate()
            .map(|(j, t)| {
                if active.map(|a| !a[ci * 4096 + j]).unwrap_or(false) {
                    return Vec::new();
                }
                let group = old.and_then(|o| cands.get(o[ci * 4096 + j] as usize)).map(|c| c.group);
                let mut v: Vec<(u32, f32)> = r
                    .query(cands, t, k)
                    .into_iter()
                    .filter(|&c| {
                        let g = cands[c as usize].group;
                        Some(g) != group && !excluded_group.get(g as usize).copied().unwrap_or(false)
                    })
                    .map(|c| (c, score(s.method, t, &cands[c as usize])))
                    .collect();
                v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                v.truncate(keep);
                v.shrink_to_fit();
                v
            })
            .collect();
        out.extend(part);
        job.report("finding candidates", ((ci + 1) * 4096).min(n) as u64, n as u64);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn assign(
    pool: &Pool,
    cands: &[Candidate],
    r: &Retrieval,
    target: &Target,
    grid: &GridSpec,
    s: &MatchSettings,
    active: &[bool],
    lists: &[Vec<(u32, f32)>],
    old: &[u32],
    excluded_group: &[bool],
    previous: Option<(&[TileAssign], &[bool], bool)>,
    stats: &mut MatchStats,
    job: &Job,
) -> Result<Vec<TileAssign>> {
    let n = grid.tiles();
    let cols = grid.cols as i64;
    let rows = grid.rows as i64;
    let cap = s.max_reuse.unwrap_or(u32::MAX);
    let mut uses = vec![0u32; pool.group_count as usize];
    let mut placed: Vec<u32> = vec![u32::MAX; n]; // candidate index
    for t in 0..n {
        if !active[t] && old[t] != u32::MAX {
            placed[t] = old[t];
            uses[cands[old[t] as usize].group as usize] += 1;
        }
    }
    let radius = s.repeat_radius as i64;
    let use_neighbors = s.repeat_penalty > 0.0 && radius > 0;
    let mut exhausted_groups = 0usize;
    let allow_keep_old = previous.is_some_and(|(_, _, allow)| allow);
    if !s.must_include.is_empty() && previous.is_none_or(|(_, _, allow)| allow) {
        let mut seen = std::collections::HashSet::new();
        let mut must_photos = Vec::new();
        for hex_hash in &s.must_include {
            let h = crate::catalog::unhex(hex_hash)?;
            if !seen.insert(h) {
                continue;
            }
            let Some(pi) = pool.index_of(&h) else {
                bail!("a must-include photo is not available in the selected photo pool; check your photo filters or folders");
            };
            let g = pool.photos[pi].group as usize;
            if excluded_group.get(g).copied().unwrap_or(false) {
                continue;
            }
            if uses[g] > 0 {
                // Already present via a pinned tile.
                continue;
            }
            must_photos.push(pi as u32);
        }
        let active_count = active.iter().filter(|&&a| a).count();
        if must_photos.len() > active_count {
            bail!("there are more must-include photos ({}) than active tiles ({active_count})", must_photos.len());
        }
        for photo_idx in must_photos {
            job.check()?;
            let g = pool.photos[photo_idx as usize].group as usize;
            if uses[g] >= cap {
                continue;
            }
            let photo_cands: Vec<u32> = cands.iter().enumerate().filter(|(_, c)| c.photo == photo_idx).map(|(ci, _)| ci as u32).collect();
            if photo_cands.is_empty() {
                continue;
            }
            let mut best: Option<(usize, u32, f32)> = None;
            let mut fallback: Option<(usize, u32, f32)> = None;
            for t in 0..n {
                if !active[t] || placed[t] != u32::MAX {
                    continue;
                }
                let same_as_old = old[t] != u32::MAX && cands[old[t] as usize].group as usize == g;
                let mut neigh_pen = 0.0f32;
                if use_neighbors {
                    let (tx, ty) = ((t as i64) % cols, (t as i64) / cols);
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let (x, y) = (tx + dx, ty + dy);
                            if (dx == 0 && dy == 0) || x < 0 || y < 0 || x >= cols || y >= rows {
                                continue;
                            }
                            let p = placed[(y * cols + x) as usize];
                            if p != u32::MAX && cands[p as usize].group as usize == g {
                                let d = dx.abs().max(dy.abs()) as f32;
                                neigh_pen += s.repeat_penalty * (radius as f32 + 1.0 - d) / radius as f32;
                            }
                        }
                    }
                }
                for &ci in &photo_cands {
                    let cost = score(s.method, &target.tiles[t], &cands[ci as usize]) + neigh_pen;
                    let slot = if same_as_old { &mut fallback } else { &mut best };
                    if slot.map(|b| cost < b.2).unwrap_or(true) {
                        *slot = Some((t, ci, cost));
                    }
                }
            }
            if let Some((t, ci, _)) = best.or(fallback) {
                placed[t] = ci;
                uses[g] += 1;
                if uses[g] == cap {
                    exhausted_groups += 1;
                }
            }
        }
    }
    let mut order: Vec<u32> = (0..n as u32).filter(|&t| active[t as usize] && placed[t as usize] == u32::MAX).collect();
    let mut rng = SplitMix64::new(s.seed);
    if s.focal_priority {
        let computed;
        let prio: &[f32] = if target.tile_priority.len() == n {
            &target.tile_priority
        } else {
            computed = tile_salience(target, &[]);
            &computed
        };
        let mut scored: Vec<(u32, f32)> = order
            .into_iter()
            .map(|t| {
                let jitter = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * 0.05;
                (t, prio.get(t as usize).copied().unwrap_or(0.0) + jitter)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        order = scored.into_iter().map(|(t, _)| t).collect();
    } else {
        rng.shuffle(&mut order);
    }
    let usage_pen = if s.method == Method::Variety { s.usage_penalty } else { 0.0 };
    let mut neigh: Vec<(u32, f32)> = Vec::new();
    let mut avail: Option<AvailIndex> = None;
    let _ = r;
    for (step, &t) in order.iter().enumerate() {
        if step % 8192 == 0 {
            job.check()?;
            job.report("assigning tiles", step as u64, order.len() as u64);
        }
        let t = t as usize;
        let old_group = (old[t] != u32::MAX).then(|| cands[old[t] as usize].group);
        neigh.clear();
        if use_neighbors {
            let (tx, ty) = ((t as i64) % cols, (t as i64) / cols);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let (x, y) = (tx + dx, ty + dy);
                    if (dx == 0 && dy == 0) || x < 0 || y < 0 || x >= cols || y >= rows {
                        continue;
                    }
                    let p = placed[(y * cols + x) as usize];
                    if p != u32::MAX {
                        let d = dx.abs().max(dy.abs()) as f32;
                        let w = s.repeat_penalty * (radius as f32 + 1.0 - d) / radius as f32;
                        neigh.push((cands[p as usize].group, w));
                    }
                }
            }
        }
        let cost_of = |ci: u32, base: f32, uses: &[u32]| -> Option<f32> {
            let g = cands[ci as usize].group;
            if Some(g) == old_group {
                return None;
            }
            if excluded_group.get(g as usize).copied().unwrap_or(false) || uses[g as usize] >= cap {
                return None;
            }
            let mut c = base + usage_pen * uses[g as usize] as f32;
            for &(ng, w) in &neigh {
                if ng == g {
                    c += w;
                }
            }
            Some(c)
        };
        let pick = |list: &[(u32, f32)], uses: &[u32]| -> Option<(u32, f32)> {
            let mut best: Option<(u32, f32)> = None;
            for &(ci, base) in list {
                if let Some(c) = cost_of(ci, base, uses) {
                    if best.map(|b| c < b.1).unwrap_or(true) {
                        best = Some((ci, c));
                    }
                }
            }
            best
        };
        let mut choice = pick(&lists[t], &uses);
        if choice.is_none() {
            // All retrieved candidates are at their reuse limit: search an index of
            // candidates that still have capacity (rebuilt when it gets stale), so
            // a narrow first search never causes a false "infeasible".
            stats.widened_searches += 1;
            for _attempt in 0..3 {
                let stale = avail.as_ref().map(|a: &AvailIndex| exhausted_groups > a.exhausted_at + a.size / 10 + 16).unwrap_or(true);
                if stale {
                    avail = AvailIndex::build(cands, s.method, &uses, excluded_group, cap, exhausted_groups)?;
                    stats.index_rebuilds += 1;
                }
                let Some(a) = &avail else { break };
                let wide: Vec<(u32, f32)> = a
                    .query(cands, &target.tiles[t], (s.candidates.max(16) * 4).min(a.size))
                    .into_iter()
                    .map(|c| (c, score(s.method, &target.tiles[t], &cands[c as usize])))
                    .collect();
                choice = pick(&wide, &uses);
                if choice.is_some() {
                    break;
                }
                avail = None; // force a rebuild
            }
        }
        // A single edit or exclusion must use a different photo. A whole selection may
        // retain an unexcluded tile when alternatives have exhausted their reuse limits.
        if choice.is_none() && previous.is_some() {
            choice = cands.iter().enumerate().filter_map(|(ci, c)| {
                cost_of(ci as u32, score(s.method, &target.tiles[t], c), &uses).map(|cost| (ci as u32, cost))
            }).min_by(|a, b| a.1.total_cmp(&b.1));
            if choice.is_none() && allow_keep_old {
                let og = cands[old[t] as usize].group as usize;
                if !excluded_group.get(og).copied().unwrap_or(false) && uses[og] < cap {
                    choice = Some((old[t], 0.0));
                }
            }
        }
        let Some((ci, _)) = choice else {
            if previous.is_some() {
                bail!("no different photo is available within the reuse limit; add photos or raise the maximum reuse");
            }
            // Cannot happen after check_feasible; keep a clear error regardless.
            bail!("internal: no photo with remaining reuse capacity for tile {t}");
        };
        let g = cands[ci as usize].group as usize;
        uses[g] += 1;
        if uses[g] == cap {
            exhausted_groups += 1;
        }
        placed[t] = ci;
    }
    // Statistics and optional colour correction.
    let mut out = Vec::with_capacity(n);
    let mut de_sum = 0f64;
    let mut cost_sum = 0f64;
    for t in 0..n {
        if placed[t] == u32::MAX {
            out.push(TileAssign::EMPTY);
            continue;
        }
        let c = &cands[placed[t] as usize];
        let tm = mean_lab(&target.tiles[t]);
        de_sum += de2(&tm, &c.mean).sqrt() as f64;
        cost_sum += score(s.method, &target.tiles[t], c) as f64;
        let adjust = if s.color_correction > 0.0 {
            let mut a = [0f32; 3];
            for k in 0..3 {
                a[k] = (tm[k] - c.mean[k]) * s.color_correction;
            }
            // Bound the shift to at most 15 ΔE.
            let m = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if m > 15.0 {
                for v in a.iter_mut() {
                    *v *= 15.0 / m;
                }
            }
            a
        } else {
            [0.0; 3]
        };
        out.push(if let Some((tiles, _, _)) = previous.filter(|_| !active[t]) { tiles[t] } else { TileAssign { photo: c.photo, crop: c.crop, adjust } });
    }
    let mut per_group = vec![0u32; pool.group_count as usize];
    for p in placed.iter().filter(|&&p| p != u32::MAX) {
        per_group[cands[*p as usize].group as usize] += 1;
    }
    stats.distinct_used = per_group.iter().filter(|&&u| u > 0).count();
    stats.max_uses = per_group.iter().copied().max().unwrap_or(0);
    let m = stats.tiles.max(1) as f64;
    stats.mean_delta_e = (de_sum / m) as f32;
    stats.mean_cost = (cost_sum / m) as f32;
    stats.nearby_repeats = count_nearby_repeats(&out, pool, grid.cols, 1);
    Ok(out)
}

/// k-d tree over the candidates whose photo group still has reuse capacity.
struct AvailIndex {
    retrieval: Retrieval,
    map: Vec<u32>,
    size: usize,
    exhausted_at: usize,
}

impl AvailIndex {
    fn build(cands: &[Candidate], method: Method, uses: &[u32], excluded_group: &[bool], cap: u32, exhausted: usize) -> Result<Option<Self>> {
        let is_avail = |c: &Candidate| !excluded_group.get(c.group as usize).copied().unwrap_or(false) && uses[c.group as usize] < cap;
        let sub: Vec<Candidate> = cands.iter().filter(|c| is_avail(c)).cloned().collect();
        let map: Vec<u32> = cands.iter().enumerate().filter(|(_, c)| is_avail(c)).map(|(i, _)| i as u32).collect();
        if sub.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { retrieval: Retrieval::build(&sub, method)?, size: sub.len(), map, exhausted_at: exhausted }))
    }
    fn query(&self, _cands: &[Candidate], t: &TileDesc, k: usize) -> Vec<u32> {
        // Brute retrieval is never used here, so the subset is not needed at query time.
        let dummy: [Candidate; 0] = [];
        let k = k.clamp(1, self.size);
        let ids = match &self.retrieval {
            Retrieval::Brute { .. } => Vec::new(),
            r => r.query_k(&dummy, t, k),
        };
        ids.into_iter().map(|i| self.map[i as usize]).collect()
    }
}

/// Number of tile pairs within `radius` (Chebyshev) that show the same photo group.
pub fn count_nearby_repeats(tiles: &[TileAssign], pool: &Pool, cols: u32, radius: i64) -> usize {
    let cols = cols as i64;
    let rows = tiles.len() as i64 / cols;
    let mut n = 0;
    for y in 0..rows {
        for x in 0..cols {
            let t = &tiles[(y * cols + x) as usize];
            if t.is_empty() {
                continue;
            }
            let g = pool.photos[t.photo as usize].group;
            for dy in 0..=radius {
                for dx in -radius..=radius {
                    if dy == 0 && dx <= 0 {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || nx >= cols || ny >= rows {
                        continue;
                    }
                    let o = &tiles[(ny * cols + nx) as usize];
                    if !o.is_empty() && pool.photos[o.photo as usize].group == g {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::PoolRow;
    use crate::faces::{cuts_face, Face};

    fn photo(w: u32, h: u32, faces: Vec<Face>) -> Pool {
        Pool::from_rows(vec![PoolRow { hash: [1; 32], width: w, height: h, grid: vec![128; descriptor::GRID_BYTES], dhash: 0, faces }], false)
    }

    #[test]
    fn face_crop_keeps_a_low_face_and_drops_crops_that_cut_it() {
        // 1:2 portrait, square tiles, face in the lower half.
        let f = Face { rect: [0.3, 0.7, 0.3, 0.15], score: 1.0 };
        let pool = photo(500, 1000, vec![f]);
        let pos = photo_crop_positions(&pool.photos[0], 1.0, true, true);
        assert!(!pos.is_empty());
        for t in &pos {
            assert!(!cuts_face(crop_rect(0.5, 1.0, *t), &[f]), "t={t}");
        }
        // Off: the usual top-biased positions, which miss this face.
        assert_eq!(photo_crop_positions(&pool.photos[0], 1.0, true, false), vec![descriptor::TALL_CROP_T, 0.0]);
        // No faces: unchanged.
        let plain = photo(500, 1000, vec![]);
        assert_eq!(photo_crop_positions(&plain.photos[0], 1.0, false, true), vec![descriptor::TALL_CROP_T]);
    }
}
