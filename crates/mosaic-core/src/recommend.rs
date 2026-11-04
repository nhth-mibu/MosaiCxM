//! Motif suggestions for the currently selected libraries and settings.
//!
//! Two passes:
//! 1. Cheap screening of every candidate from a small image (stored
//!    thumbnail for library photos, a reduced decode for external files):
//!    colour coverage by the pool, repetition pressure, contrast and the share
//!    of detail that will disappear at the chosen grid density.
//! 2. Low-resolution trial mosaics (real matching on descriptors) for the
//!    strongest candidates, measuring achieved colour error, edge
//!    preservation and variety.
//!
//! The scores estimate *technical* suitability for this collection and these
//! settings. They do not measure artistic quality.
//!
//! A library photo (or an external file whose content is in the pool) is
//! never allowed to match itself: its content hash, its duplicate group and
//! near-identical dHashes are removed from the pool before evaluation.

use crate::catalog::{Catalog, Hash};
use crate::color::{de2, Lab};
use crate::decode;
use crate::descriptor::{self, mean_lab, CELLS};
use crate::job::Job;
use crate::matching::{match_tiles, GridSpec, MatchSettings, Target};
use crate::pool::Pool;
use crate::rng::SplitMix64;
use anyhow::Result;
use image::RgbImage;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::num::NonZero;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub enum MotifSource {
    File(PathBuf),
    Library { hash: String },
}

#[derive(Debug, Clone)]
pub struct MotifCandidate {
    pub label: String,
    pub source: MotifSource,
    hash: Option<Hash>,
}

impl MotifCandidate {
    pub fn file(p: &Path) -> Self {
        Self { label: p.display().to_string(), source: MotifSource::File(p.to_path_buf()), hash: None }
    }
    pub fn from_folder(dir: &Path) -> Result<Vec<Self>> {
        let mut v: Vec<Self> = crate::indexer::walk_images(dir)
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                e.path().extension().and_then(|x| x.to_str()).map(|x| crate::indexer::IMAGE_EXTENSIONS.contains(&x.to_ascii_lowercase().as_str())).unwrap_or(false)
            })
            .map(|e| Self::file(e.path()))
            .collect();
        v.sort_by(|a, b| a.label.cmp(&b.label));
        Ok(v)
    }
    /// A seeded sample of `n` pool photos.
    pub fn from_pool(pool: &Pool, n: usize, seed: u64) -> Vec<Self> {
        let mut idx: Vec<usize> = (0..pool.photos.len()).collect();
        SplitMix64::new(seed ^ 0x5EED).shuffle(&mut idx);
        idx.truncate(n);
        idx.sort_unstable();
        idx.into_iter()
            .map(|i| {
                let h = pool.photos[i].hash;
                Self { label: format!("library photo {}", &crate::catalog::hex(&h)[..12]), source: MotifSource::Library { hash: crate::catalog::hex(&h) }, hash: Some(h) }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct RecommendOptions {
    pub trials: usize,
    pub preview_dir: Option<PathBuf>,
    pub trial_max_tiles: usize,
    pub preview_tile_px: u32,
}

impl Default for RecommendOptions {
    fn default() -> Self {
        Self { trials: 8, preview_dir: None, trial_max_tiles: 6000, preview_tile_px: 12 }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ScreenScores {
    /// Share of motif areas with a pool photo within ΔE 10.
    pub coverage: f32,
    pub mean_nearest_de: f32,
    /// Share of tiles that cannot get a close photo without exceeding the reuse level.
    pub repetition_pressure: f32,
    pub contrast: f32,
    /// Share of the motif's luminance variation finer than one tile (lost).
    pub detail_loss: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TrialScores {
    pub cols: u32,
    pub rows: u32,
    pub mean_delta_e: f32,
    pub edge_correlation: f32,
    pub distinct_share: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub label: String,
    pub source: MotifSource,
    pub score: f32,
    pub screen: ScreenScores,
    pub trial: Option<TrialScores>,
    pub notes: Vec<String>,
    pub preview: Option<PathBuf>,
    pub excluded_self: bool,
    /// Small picture of the candidate itself (long side at most
    /// [`PICTURE_THUMB_PX`]), so every row can be shown even when no trial
    /// mosaic was rendered. `None` only when the picture could not be read.
    #[serde(skip)]
    pub picture: Option<RgbImage>,
}

/// Long side (px) of [`Recommendation::picture`].
pub const PICTURE_THUMB_PX: u32 = 200;

fn picture_thumb(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let m = w.max(h);
    if m <= PICTURE_THUMB_PX {
        return img.clone();
    }
    let s = PICTURE_THUMB_PX as f64 / m as f64;
    image::imageops::thumbnail(img, ((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1))
}

struct Loaded {
    small: RgbImage,
    hash: Option<Hash>,
    dhash: Option<u64>,
}

/// Catalogue data needed for a library candidate (read on one thread; SQLite
/// connections are not shared between threads).
type Prefetched = Option<(Vec<u8>, Option<u64>)>;

fn prefetch(cat: &Catalog, c: &MotifCandidate) -> Result<Prefetched> {
    match (&c.source, c.hash) {
        (MotifSource::Library { .. }, Some(h)) => {
            let t = cat.thumb(&h)?.ok_or_else(|| anyhow::anyhow!("no thumbnail"))?;
            Ok(Some((t, cat.descriptor(&h)?.map(|r| r.dhash))))
        }
        _ => Ok(None),
    }
}

fn load_small(pre: Prefetched, c: &MotifCandidate) -> Result<Loaded> {
    match (&c.source, c.hash, pre) {
        (MotifSource::Library { .. }, Some(h), Some((t, d))) => Ok(Loaded { small: decode::decode_bytes(&t, None)?.image, hash: Some(h), dhash: d }),
        (MotifSource::File(p), _, _) => {
            let bytes = std::fs::read(p)?;
            let d = decode::decode_bytes(&bytes, Some(256))?;
            let desc = descriptor::compute(&d.image, d.info.full_width, d.info.full_height);
            Ok(Loaded { small: d.image, hash: Some(*blake3::hash(&bytes).as_bytes()), dhash: Some(desc.dhash) })
        }
        _ => anyhow::bail!("invalid candidate"),
    }
}

fn screen_grid(grid: &GridSpec, max_side: u32) -> GridSpec {
    let s = (max_side as f64 / grid.cols.max(grid.rows) as f64).min(1.0);
    GridSpec { cols: ((grid.cols as f64 * s).round() as u32).max(1), rows: ((grid.rows as f64 * s).round() as u32).max(1), ..*grid }
}

/// Supply of distinct photo groups per 10-ΔE Lab bin (with neighbouring bins).
struct Supply {
    bins: HashMap<(i32, i32, i32), u32>,
}

fn bin(l: &Lab) -> (i32, i32, i32) {
    ((l[0] / 10.0).floor() as i32, (l[1] / 10.0).floor() as i32, (l[2] / 10.0).floor() as i32)
}

impl Supply {
    fn new(pool: &Pool) -> Self {
        let mut raw: HashMap<(i32, i32, i32), std::collections::HashSet<u32>> = HashMap::new();
        for p in &pool.photos {
            raw.entry(bin(&p.mean)).or_default().insert(p.group);
        }
        let mut bins = HashMap::new();
        for &k in raw.keys() {
            for dl in -1..=1 {
                for da in -1..=1 {
                    for db in -1..=1 {
                        let n = (k.0 + dl, k.1 + da, k.2 + db);
                        *bins.entry(n).or_insert(0) += raw[&k].len() as u32;
                    }
                }
            }
        }
        Self { bins }
    }
    fn at(&self, l: &Lab) -> u32 {
        *self.bins.get(&bin(l)).unwrap_or(&0)
    }
}

pub fn describe_color(l: &Lab) -> String {
    let chroma = (l[1] * l[1] + l[2] * l[2]).sqrt();
    let light = if l[0] < 30.0 { "dark " } else if l[0] > 75.0 { "bright " } else { "" };
    if chroma < 15.0 {
        return format!("{}{}", light, if l[0] < 30.0 { "shadow" } else if l[0] > 75.0 { "white/light grey" } else { "grey" }).trim().to_string();
    }
    let h = l[2].atan2(l[1]).to_degrees().rem_euclid(360.0);
    let hue = match h as u32 {
        0..=20 | 345..=360 => "magenta-red",
        21..=50 => "red",
        51..=75 => "orange",
        76..=105 => "yellow",
        106..=165 => "green",
        166..=230 => "cyan",
        231..=290 => "blue",
        _ => "purple",
    };
    format!("{light}{hue}")
}

pub fn recommend(cat: &Catalog, pool: &Pool, cands: Vec<MotifCandidate>, grid: &GridSpec, s: &MatchSettings, opts: &RecommendOptions, job: &Job) -> Result<Vec<Recommendation>> {
    let means: Vec<[f32; 3]> = pool.photos.iter().map(|p| p.mean).collect();
    let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&means).map_err(|e| anyhow::anyhow!("kd-tree: {e:?}"))?;
    let supply = Supply::new(pool);
    let sg = screen_grid(grid, 24);
    let reuse_level = s.max_reuse.unwrap_or(4).max(1) as f32;
    let full_tiles = grid.tiles() as f32;
    let n = cands.len();
    job.report("screening motifs", 0, n as u64);
    let done = std::sync::atomic::AtomicU64::new(0);
    let pre: Vec<(MotifCandidate, Result<Prefetched>)> = cands.into_iter().map(|c| { let p = prefetch(cat, &c); (c, p) }).collect();
    let mut screened: Vec<(MotifCandidate, Option<Loaded>, ScreenScores, Vec<String>)> = pre
        .into_par_iter()
        .map(|(c, p)| {
            let r = p.and_then(|p| load_small(p, &c)).map(|ld| {
                let (sc, notes) = screen(&ld, pool, &tree, &supply, &sg, grid, reuse_level, full_tiles, s.max_reuse.is_none());
                (ld, sc, notes)
            });
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            job.report("screening motifs", k, n as u64);
            match r {
                Ok((ld, sc, notes)) => (c, Some(ld), sc, notes),
                Err(e) => (c, None, ScreenScores::default(), vec![format!("could not be read: {e}")]),
            }
        })
        .collect();
    job.check()?;
    screened.sort_by(|a, b| b.2.score.total_cmp(&a.2.score).then(a.0.label.cmp(&b.0.label)));
    let libs: Vec<i64> = vec![];
    let _ = libs;
    let mut out = Vec::new();
    let trials = opts.trials.min(screened.len());
    for (i, (c, ld, sc, mut notes)) in screened.into_iter().enumerate() {
        let mut rec = Recommendation { label: c.label.clone(), source: c.source.clone(), score: sc.score, screen: sc.clone(), trial: None, notes: vec![], preview: None, excluded_self: false, picture: ld.as_ref().map(|l| picture_thumb(&l.small)) };
        if let (true, Some(ld)) = (i < trials, ld.as_ref()) {
            job.check()?;
            job.report("trial mosaics", i as u64, trials as u64);
            let in_pool = ld.hash.map(|h| pool.index_of(&h).is_some()).unwrap_or(false);
            let near = ld.dhash.map(|d| pool.photos.iter().any(|p| descriptor::hamming(p.dhash, d) <= 4)).unwrap_or(false);
            let sub;
            let tpool = if in_pool || near {
                rec.excluded_self = true;
                sub = pool.excluding(&ld.hash.unwrap_or([0; 32]), ld.dhash, s.merge_near_duplicates);
                &sub
            } else {
                pool
            };
            match trial(cat, &c, ld, tpool, grid, s, opts, i, job) {
                Ok((ts, prev, tnotes)) => {
                    rec.score = 0.4 * sc.score + 0.6 * ts.score;
                    rec.trial = Some(ts);
                    rec.preview = prev;
                    notes.extend(tnotes);
                }
                Err(e) => notes.push(format!("trial mosaic failed: {e}")),
            }
            if rec.excluded_self {
                notes.push("this photo and its duplicates were excluded from the sources".into());
            }
        }
        rec.notes = notes;
        out.push(rec);
    }
    out.sort_by(|a, b| (b.trial.is_some(), b.score).partial_cmp(&(a.trial.is_some(), a.score)).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn screen(
    ld: &Loaded,
    pool: &Pool,
    tree: &ImmutableKdTree<f32, 3>,
    supply: &Supply,
    sg: &GridSpec,
    grid: &GridSpec,
    reuse_level: f32,
    full_tiles: f32,
    unlimited: bool,
) -> (ScreenScores, Vec<String>) {
    let (target, _) = Target::from_image(&ld.small, sg);
    let tile_means: Vec<Lab> = target.tiles.iter().map(mean_lab).collect();
    // Coverage: nearest pool photo per tile, skipping the motif itself.
    let excluded = |i: u32| {
        let p = &pool.photos[i as usize];
        Some(p.hash) == ld.hash || ld.dhash.map(|d| descriptor::hamming(d, p.dhash) <= 4).unwrap_or(false)
    };
    let mut nn_sum = 0f32;
    let mut covered = 0usize;
    let k = NonZero::new(8.min(pool.photos.len()).max(1)).unwrap();
    for m in &tile_means {
        let r = tree.query(m).nearest_n::<SquaredEuclidean<f32>>(k).execute();
        let d = r.iter().find(|x| !excluded(x.item)).map(|x| x.distance.sqrt()).unwrap_or(50.0);
        nn_sum += d;
        if d < 10.0 {
            covered += 1;
        }
    }
    let nt = tile_means.len() as f32;
    let mean_nn = nn_sum / nt;
    let coverage = covered as f32 / nt;
    // Repetition pressure per colour bin at full grid size.
    let mut demand: HashMap<(i32, i32, i32), (f32, Lab)> = HashMap::new();
    for m in &tile_means {
        let e = demand.entry(bin(m)).or_insert((0.0, *m));
        e.0 += full_tiles / nt;
    }
    let mut over = 0f32;
    let mut worst: Option<(f32, Lab)> = None;
    for (_, (d, lab)) in demand.iter() {
        let cap = supply.at(lab) as f32 * reuse_level;
        let deficit = (d - cap).max(0.0);
        over += deficit;
        if deficit > 0.0 && worst.map(|w| deficit > w.0).unwrap_or(true) {
            worst = Some((deficit, *lab));
        }
    }
    let pressure = (over / full_tiles).min(1.0);
    // Contrast and detail loss from luminance.
    let ls: Vec<f32> = tile_means.iter().map(|m| m[0]).collect();
    let mean_l = ls.iter().sum::<f32>() / nt;
    let std_l = (ls.iter().map(|l| (l - mean_l).powi(2)).sum::<f32>() / nt).sqrt();
    let contrast = (std_l / 25.0).min(1.0);
    let detail_loss = detail_loss(&ld.small, grid);
    let cov_score = 0.5 * coverage + 0.5 * (-mean_nn / 12.0).exp();
    let score = 0.35 * cov_score + 0.25 * (1.0 - pressure) + 0.2 * contrast + 0.2 * (1.0 - detail_loss);
    let mut notes = Vec::new();
    notes.push(match coverage {
        c if c > 0.9 => "strong colour coverage".to_string(),
        c if c > 0.6 => "good colour coverage".to_string(),
        c => format!("weak colour coverage ({:.0}% of areas have a close photo)", c * 100.0),
    });
    if let Some((_, lab)) = worst.filter(|_| pressure > 0.05) {
        notes.push(format!(
            "limited {} photos may cause {}",
            describe_color(&lab),
            if unlimited { "visible repetition" } else { "weaker matches under the reuse limit" }
        ));
    }
    if contrast < 0.35 {
        notes.push("low contrast; the motif may be hard to recognise".into());
    } else if contrast > 0.7 {
        notes.push("strong light/dark contrast".into());
    }
    if detail_loss > 0.45 {
        notes.push(format!("much fine detail ({:.0}%) is smaller than a tile and will disappear; use more tiles or view from further away", detail_loss * 100.0));
    } else if detail_loss < 0.2 {
        notes.push("major shapes survive at this grid density".into());
    }
    (ScreenScores { coverage, mean_nearest_de: mean_nn, repetition_pressure: pressure, contrast, detail_loss, score }, notes)
}

/// Fraction of luminance variance below tile scale.
fn detail_loss(img: &RgbImage, grid: &GridSpec) -> f32 {
    let mosaic_aspect = (grid.width_mm / grid.height_mm) as f32;
    let crop = descriptor::crop_rect(img.width() as f32 / img.height() as f32, mosaic_aspect, 0.5);
    let fine_w = ((crop[2] * img.width() as f32) as usize).clamp(1, grid.cols as usize * CELLS * 2);
    let fine_h = ((crop[3] * img.height() as f32) as usize).clamp(1, grid.rows as usize * CELLS * 2);
    let x = (crop[0] * img.width() as f32) as u32;
    let y = (crop[1] * img.height() as f32) as u32;
    let sub = image::imageops::crop_imm(img, x, y, ((crop[2] * img.width() as f32) as u32).max(1), ((crop[3] * img.height() as f32) as u32).max(1)).to_image();
    let fine = descriptor::area_average_lab(&sub, fine_w, fine_h);
    let lf: Vec<f32> = fine.iter().map(|c| c[0]).collect();
    let mean = lf.iter().sum::<f32>() / lf.len() as f32;
    let total: f32 = lf.iter().map(|l| (l - mean).powi(2)).sum();
    if total < 1e-3 {
        return 0.0;
    }
    let mut resid = 0f32;
    for (i, l) in lf.iter().enumerate() {
        let (fx, fy) = (i % fine_w, i / fine_w);
        let tx = fx * grid.cols as usize / fine_w;
        let ty = fy * grid.rows as usize / fine_h;
        // Mean of this tile's fine cells.
        let x0 = tx * fine_w / grid.cols as usize;
        let x1 = ((tx + 1) * fine_w / grid.cols as usize).max(x0 + 1);
        let y0 = ty * fine_h / grid.rows as usize;
        let y1 = ((ty + 1) * fine_h / grid.rows as usize).max(y0 + 1);
        let mut s = 0f32;
        let mut n = 0f32;
        for yy in y0..y1.min(fine_h) {
            for xx in x0..x1.min(fine_w) {
                s += lf[yy * fine_w + xx];
                n += 1.0;
            }
        }
        resid += (l - s / n.max(1.0)).powi(2);
    }
    (resid / total).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn trial(cat: &Catalog, c: &MotifCandidate, ld: &Loaded, pool: &Pool, grid: &GridSpec, s: &MatchSettings, opts: &RecommendOptions, rank: usize, job: &Job) -> Result<(TrialScores, Option<PathBuf>, Vec<String>)> {
    let mut notes = Vec::new();
    let tiles = grid.tiles();
    let (tg, ts) = if tiles > opts.trial_max_tiles {
        let f = (opts.trial_max_tiles as f64 / tiles as f64).sqrt();
        let g = GridSpec { cols: ((grid.cols as f64 * f).round() as u32).max(1), rows: ((grid.rows as f64 * f).round() as u32).max(1), ..*grid };
        let mut s2 = s.clone();
        s2.max_reuse = s.max_reuse.map(|m| ((m as f64 * g.tiles() as f64 / tiles as f64).ceil() as u32).max(1));
        (g, s2)
    } else {
        (*grid, s.clone())
    };
    // Motif at trial resolution: a reduced decode of the original when available.
    let img = match &c.source {
        MotifSource::File(p) => decode::decode_file(p, Some((tg.cols.max(tg.rows) as usize * CELLS) as u32))?.image,
        MotifSource::Library { .. } => ld.small.clone(),
    };
    let (target, crop) = Target::from_image(&img, &tg);
    let (assign, stats) = match match_tiles(pool, &target, &tg, &ts, job) {
        Ok(v) => v,
        Err(crate::matching::MatchError::Infeasible(inf)) => {
            notes.push(format!("not feasible with these reuse settings: {}", inf.message));
            return Ok((TrialScores { cols: tg.cols, rows: tg.rows, ..Default::default() }, None, notes));
        }
        Err(e) => return Err(anyhow::anyhow!("{e}")),
    };
    // Edge preservation: correlation of horizontal+vertical tile-mean L gradients.
    let cands_l: Vec<f32> = assign
        .iter()
        .map(|a| mean_lab(&descriptor::crop_descriptor(&pool.photos[a.photo as usize].grid, a.crop))[0])
        .collect();
    let tl: Vec<f32> = target.tiles.iter().map(|t| mean_lab(t)[0]).collect();
    let (cols, rows) = (tg.cols as usize, tg.rows as usize);
    let mut gt = Vec::new();
    let mut gm = Vec::new();
    for y in 0..rows {
        for x in 0..cols {
            if x + 1 < cols {
                gt.push(tl[y * cols + x + 1] - tl[y * cols + x]);
                gm.push(cands_l[y * cols + x + 1] - cands_l[y * cols + x]);
            }
            if y + 1 < rows {
                gt.push(tl[(y + 1) * cols + x] - tl[y * cols + x]);
                gm.push(cands_l[(y + 1) * cols + x] - cands_l[y * cols + x]);
            }
        }
    }
    let edge_corr = correlation(&gt, &gm);
    let distinct_share = stats.distinct_used as f32 / (tg.tiles().min(pool.group_count as usize).max(1)) as f32;
    let score = 0.55 * (-stats.mean_delta_e / 15.0).exp() + 0.3 * edge_corr.max(0.0) + 0.15 * distinct_share.min(1.0);
    notes.push(format!("trial {}×{}: mean colour error ΔE {:.1}, edges kept {:.0}%", tg.cols, tg.rows, stats.mean_delta_e, edge_corr.max(0.0) * 100.0));
    if edge_corr > 0.8 {
        notes.push("major edges preserved".into());
    }
    let mut preview = None;
    if let Some(dir) = &opts.preview_dir {
        std::fs::create_dir_all(dir)?;
        // Build a throwaway project for rendering the trial from thumbnails.
        let motif = crate::project::MotifRef { path: PathBuf::from(&c.label), content_hash: String::new(), width: img.width(), height: img.height(), crop };
        let p = crate::project::Project::create(pool, vec![], &target, motif, tg, ts.clone(), job).map_err(|e| anyhow::anyhow!("{e}"))?;
        let pimg = crate::render::render_preview(cat, &p, opts.preview_tile_px, job)?;
        let path = dir.join(format!("suggestion-{:02}.png", rank + 1));
        pimg.save(&path)?;
        preview = Some(path);
    }
    let _ = de2;
    Ok((TrialScores { cols: tg.cols, rows: tg.rows, mean_delta_e: stats.mean_delta_e, edge_correlation: edge_corr, distinct_share, score }, preview, notes))
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    if saa < 1e-6 || sbb < 1e-6 {
        return 0.0;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}
