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
use crate::pool::Pool;
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
    pub merge_near_duplicates: bool,
    pub seed: u64,
    /// Candidates retrieved per tile before reranking.
    pub candidates: usize,
    /// 0 = off. Fraction (≤ 0.5) of each tile's mean-colour error corrected at render time.
    pub color_correction: f32,
    /// 0 = off. Opacity (≤ 0.5) of the motif blended over the mosaic at render time.
    pub motif_blend: f32,
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
            merge_near_duplicates: true,
            seed: 1,
            candidates: 48,
            color_correction: 0.0,
            motif_blend: 0.0,
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

/// Target motif sampled at CELLS×CELLS Lab cells per tile.
#[derive(Debug, Clone)]
pub struct Target {
    pub cols: u32,
    pub rows: u32,
    pub tiles: Vec<TileDesc>,
}

impl Target {
    /// `crop` is the normalised part of the motif used (cover fit).
    pub fn from_image(img: &RgbImage, grid: &GridSpec) -> (Self, [f32; 4]) {
        let motif_aspect = img.width() as f32 / img.height() as f32;
        let mosaic_aspect = (grid.width_mm / grid.height_mm) as f32;
        let crop = crop_rect(motif_aspect, mosaic_aspect, 0.5);
        let x = (crop[0] * img.width() as f32).round() as u32;
        let y = (crop[1] * img.height() as f32).round() as u32;
        let w = ((crop[2] * img.width() as f32).round() as u32).clamp(1, img.width() - x);
        let h = ((crop[3] * img.height() as f32).round() as u32).clamp(1, img.height() - y);
        let sub = image::imageops::crop_imm(img, x, y, w, h).to_image();
        let gw = grid.cols as usize * CELLS;
        let gh = grid.rows as usize * CELLS;
        let cells = descriptor::area_average_lab(&sub, gw, gh);
        (Self::from_cells(grid.cols, grid.rows, &cells), crop)
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
        Self { cols, rows, tiles }
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

pub fn build_candidates(pool: &Pool, tile_aspect: f32, crop_offsets: bool) -> Vec<Candidate> {
    pool.photos
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, p)| {
            crop_positions(p.aspect(), tile_aspect, crop_offsets).iter().map(move |&t| {
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
        let k = k.clamp(1, cands.len().max(1));
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
    pub full_scans: usize,
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
    let tiles = grid.tiles();
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
    s.validate()?;
    if pool.photos.is_empty() {
        return Err(anyhow::anyhow!("no source photos available").into());
    }
    check_feasible(grid, pool.group_count, s).map_err(MatchError::Infeasible)?;
    let mut stats = MatchStats { pool_photos: pool.photos.len(), pool_groups: pool.group_count, tiles: grid.tiles(), ..Default::default() };
    let t0 = Instant::now();
    job.report("building candidates", 0, 1);
    let cands = build_candidates(pool, grid.tile_aspect(), s.crop_offsets);
    let retrieval = Retrieval::build(&cands, s.method)?;
    stats.candidates = cands.len();
    stats.build_ms = ms(t0);
    job.check()?;
    let lists = retrieve(&cands, &retrieval, target, s, job)?;
    stats.retrieval_ms = ms(t0) - stats.build_ms;
    let t1 = Instant::now();
    let tiles = assign(pool, &cands, &retrieval, target, grid, s, &lists, &mut stats, job)?;
    stats.assign_ms = ms(t1);
    Ok((tiles, stats))
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Per tile: retrieved candidates reranked by method score (ascending).
pub fn retrieve(cands: &[Candidate], r: &Retrieval, target: &Target, s: &MatchSettings, job: &Job) -> Result<Vec<Vec<(u32, f32)>>> {
    let k = s.candidates.max(4);
    let n = target.tiles.len();
    let mut out = Vec::with_capacity(n);
    for (ci, chunk) in target.tiles.chunks(4096).enumerate() {
        job.check()?;
        let part: Vec<Vec<(u32, f32)>> = chunk
            .par_iter()
            .map(|t| {
                let mut v: Vec<(u32, f32)> = r.query(cands, t, k).into_iter().map(|c| (c, score(s.method, t, &cands[c as usize]))).collect();
                v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
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
    lists: &[Vec<(u32, f32)>],
    stats: &mut MatchStats,
    job: &Job,
) -> Result<Vec<TileAssign>> {
    let n = grid.tiles();
    let cols = grid.cols as i64;
    let rows = grid.rows as i64;
    let cap = s.max_reuse.unwrap_or(u32::MAX);
    let mut uses = vec![0u32; pool.group_count as usize];
    let mut placed: Vec<u32> = vec![u32::MAX; n]; // candidate index
    let mut order: Vec<u32> = (0..n as u32).collect();
    SplitMix64::new(s.seed).shuffle(&mut order);
    let radius = s.repeat_radius as i64;
    let use_neighbors = s.repeat_penalty > 0.0 && radius > 0;
    let usage_pen = if s.method == Method::Variety { s.usage_penalty } else { 0.0 };
    let mut neigh: Vec<(u32, f32)> = Vec::new();
    for (step, &t) in order.iter().enumerate() {
        if step % 8192 == 0 {
            job.check()?;
            job.report("assigning tiles", step as u64, n as u64);
        }
        let t = t as usize;
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
            if uses[g as usize] >= cap {
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
            // All retrieved candidates are at their reuse limit: widen the search.
            stats.widened_searches += 1;
            let mut k = lists[t].len().max(16) * 8;
            while choice.is_none() && k < cands.len() {
                let wide: Vec<(u32, f32)> = r.query(cands, &target.tiles[t], k).into_iter().map(|c| (c, score(s.method, &target.tiles[t], &cands[c as usize]))).collect();
                choice = pick(&wide, &uses);
                k *= 8;
            }
            if choice.is_none() {
                stats.full_scans += 1;
                let all: Vec<(u32, f32)> = (0..cands.len() as u32).map(|c| (c, score(s.method, &target.tiles[t], &cands[c as usize]))).collect();
                choice = pick(&all, &uses);
            }
        }
        let Some((ci, _)) = choice else {
            // Cannot happen after check_feasible; keep a clear error regardless.
            bail!("internal: no photo with remaining reuse capacity for tile {t}");
        };
        uses[cands[ci as usize].group as usize] += 1;
        placed[t] = ci;
    }
    // Statistics and optional colour correction.
    let mut out = Vec::with_capacity(n);
    let mut de_sum = 0f64;
    let mut cost_sum = 0f64;
    for t in 0..n {
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
        out.push(TileAssign { photo: c.photo, crop: c.crop, adjust });
    }
    let mut per_group = vec![0u32; pool.group_count as usize];
    for p in &placed {
        per_group[cands[*p as usize].group as usize] += 1;
    }
    stats.distinct_used = per_group.iter().filter(|&&u| u > 0).count();
    stats.max_uses = per_group.iter().copied().max().unwrap_or(0);
    stats.mean_delta_e = (de_sum / n as f64) as f32;
    stats.mean_cost = (cost_sum / n as f64) as f32;
    stats.nearby_repeats = count_nearby_repeats(&out, pool, grid.cols, 1);
    Ok(out)
}

/// Number of tile pairs within `radius` (Chebyshev) that show the same photo group.
pub fn count_nearby_repeats(tiles: &[TileAssign], pool: &Pool, cols: u32, radius: i64) -> usize {
    let cols = cols as i64;
    let rows = tiles.len() as i64 / cols;
    let mut n = 0;
    for y in 0..rows {
        for x in 0..cols {
            let g = pool.photos[tiles[(y * cols + x) as usize].photo as usize].group;
            for dy in 0..=radius {
                for dx in -radius..=radius {
                    if dy == 0 && dx <= 0 {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || nx >= cols || ny >= rows {
                        continue;
                    }
                    if pool.photos[tiles[(ny * cols + nx) as usize].photo as usize].group == g {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}
