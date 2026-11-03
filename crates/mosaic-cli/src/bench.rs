//! Reproducible benchmarks with synthetic data.

use anyhow::Result;
use clap::Subcommand;
use mosaic_core::catalog::PoolRow;
use mosaic_core::color::{lab_to_srgb8, quantize_lab};
use mosaic_core::descriptor::{GRID, GRID_BYTES};
use mosaic_core::job::Job;
use mosaic_core::matching::{build_candidates, match_tiles, GridSpec, MatchSettings, Method, Retrieval, Target};
use mosaic_core::pool::Pool;
use mosaic_core::rng::SplitMix64;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Subcommand)]
pub enum BenchCmd {
    /// Candidate retrieval and assignment on synthetic descriptors.
    Match {
        #[arg(long, default_value_t = 100_000)]
        photos: usize,
        #[arg(long, default_value_t = 150)]
        cols: u32,
        #[arg(long, default_value_t = 150)]
        rows: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Also time an exhaustive scan for this many tiles (baseline).
        #[arg(long, default_value_t = 2000)]
        brute_tiles: usize,
    },
    /// Write synthetic JPEG photos (varied colours, gradients, shapes) for import benchmarks.
    GenPhotos {
        dir: PathBuf,
        #[arg(long, default_value_t = 1000)]
        count: usize,
        #[arg(long, default_value_t = 640)]
        width: u32,
        #[arg(long, default_value_t = 480)]
        height: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

pub fn run(_dd: &Path, cmd: BenchCmd, job: &Job) -> Result<()> {
    match cmd {
        BenchCmd::Match { photos, cols, rows, seed, brute_tiles } => bench_match(photos, cols, rows, seed, brute_tiles, job),
        BenchCmd::GenPhotos { dir, count, width, height, seed } => gen_photos(&dir, count, width, height, seed, job),
    }
}

/// Synthetic descriptor: a random smooth gradient between two random colours,
/// with random brightness bias so the pool has a natural spread.
fn synth_grid(r: &mut SplitMix64) -> Vec<u8> {
    let rnd = |r: &mut SplitMix64| (r.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
    let c0 = [rnd(r) * 100.0, rnd(r) * 160.0 - 80.0, rnd(r) * 160.0 - 80.0];
    let c1 = [rnd(r) * 100.0, rnd(r) * 160.0 - 80.0, rnd(r) * 160.0 - 80.0];
    let ang = rnd(r) * std::f32::consts::TAU;
    let (dx, dy) = (ang.cos(), ang.sin());
    let mut g = Vec::with_capacity(GRID_BYTES);
    for y in 0..GRID {
        for x in 0..GRID {
            let t = (((x as f32 / 15.0 - 0.5) * dx + (y as f32 / 15.0 - 0.5) * dy) + 0.7).clamp(0.0, 1.0);
            let c = [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
            g.extend_from_slice(&quantize_lab(c));
        }
    }
    g
}

pub fn synthetic_pool(n: usize, seed: u64) -> Pool {
    let mut r = SplitMix64::new(seed);
    let rows = (0..n)
        .map(|i| {
            let mut hash = [0u8; 32];
            hash[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let (w, h) = if r.below(3) == 0 { (3000, 4000) } else { (4000, 3000) };
            PoolRow { hash, width: w, height: h, grid: synth_grid(&mut r), dhash: r.next_u64() }
        })
        .collect();
    Pool::from_rows(rows, true)
}

fn bench_match(photos: usize, cols: u32, rows: u32, seed: u64, brute_tiles: usize, job: &Job) -> Result<()> {
    println!("Hardware: {} logical CPUs ({}), rayon threads {}", num_cpus(), std::env::consts::ARCH, rayon::current_num_threads());
    println!("Dataset: {photos} synthetic descriptors (random two-colour gradients), target {cols}×{rows} = {} tiles (synthetic gradient target)", cols * rows);
    let t = Instant::now();
    let pool = synthetic_pool(photos, seed);
    println!("pool build: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let grid = GridSpec::new(1500.0, 1500.0, cols, rows)?;
    // Target: smooth synthetic image cells.
    let gw = cols as usize * 4;
    let gh = rows as usize * 4;
    let cells: Vec<[f32; 3]> = (0..gw * gh)
        .map(|i| {
            let (x, y) = ((i % gw) as f32 / gw as f32, (i / gw) as f32 / gh as f32);
            [20.0 + 70.0 * y, 60.0 * (x * 6.0).sin(), 60.0 * (y * 4.0 + x).cos()]
        })
        .collect();
    let _ = lab_to_srgb8;
    let target = Target::from_cells(cols, rows, &cells);
    for method in [Method::Classic, Method::Structure] {
        let s = MatchSettings::for_method(method);
        let t = Instant::now();
        let cands = build_candidates(&pool, grid.tile_aspect(), s.crop_offsets);
        let build_c = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let r = Retrieval::build(&cands, method)?;
        let build_t = t.elapsed().as_secs_f64() * 1000.0;
        let n = brute_tiles.min(target.tiles.len());
        let t = Instant::now();
        let kd: Vec<Vec<u32>> = target.tiles[..n].iter().map(|tt| r.query(&cands, tt, s.candidates)).collect();
        let kd_ms = t.elapsed().as_secs_f64() * 1000.0;
        let brute = Retrieval::Brute { layout: method != Method::Classic };
        let t = Instant::now();
        let bf: Vec<Vec<u32>> = target.tiles[..n].iter().map(|tt| brute.query(&cands, tt, s.candidates)).collect();
        let bf_ms = t.elapsed().as_secs_f64() * 1000.0;
        let agree = kd.iter().zip(&bf).filter(|(a, b)| a.first() == b.first()).count();
        println!(
            "{:<9} candidates {} (build {:.0} ms), kd-tree build {:.0} ms | {} tiles single-thread: kd-tree {:.2} ms/tile, exhaustive {:.2} ms/tile ({:.0}× faster), same best candidate {:.1}%",
            method.name(), cands.len(), build_c, build_t, n, kd_ms / n as f64, bf_ms / n as f64, bf_ms / kd_ms, agree as f64 * 100.0 / n as f64
        );
    }
    for (label, method, reuse) in [
        ("classic, unlimited", Method::Classic, None),
        ("structure, unlimited", Method::Structure, None),
        ("structure, max 1", Method::Structure, Some(1)),
        ("variety, max 2", Method::Variety, Some(2)),
    ] {
        let mut s = MatchSettings::for_method(method);
        s.max_reuse = reuse;
        let t = Instant::now();
        match match_tiles(&pool, &target, &grid, &s, job) {
            Ok((_, st)) => println!(
                "full match {label:<22}: total {:.0} ms (candidates+index {:.0}, retrieval {:.0}, assignment {:.0}); distinct {}, max uses {}, ΔE {:.1}, widened {}, full scans {}",
                t.elapsed().as_secs_f64() * 1000.0, st.build_ms, st.retrieval_ms, st.assign_ms, st.distinct_used, st.max_uses, st.mean_delta_e, st.widened_searches, st.full_scans
            ),
            Err(e) => println!("full match {label}: {e}"),
        }
    }
    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn gen_photos(dir: &Path, count: usize, w: u32, h: u32, seed: u64, job: &Job) -> Result<()> {
    use rayon::prelude::*;
    std::fs::create_dir_all(dir)?;
    let t = Instant::now();
    (0..count).into_par_iter().try_for_each(|i| -> Result<()> {
        let mut r = SplitMix64::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64));
        let rnd = |r: &mut SplitMix64| (r.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        let c0 = [rnd(&mut r) * 255.0, rnd(&mut r) * 255.0, rnd(&mut r) * 255.0];
        let c1 = [rnd(&mut r) * 255.0, rnd(&mut r) * 255.0, rnd(&mut r) * 255.0];
        let c2 = [rnd(&mut r) * 255.0, rnd(&mut r) * 255.0, rnd(&mut r) * 255.0];
        let ang = rnd(&mut r) * std::f32::consts::TAU;
        let (cx, cy, rad) = (rnd(&mut r) * w as f32, rnd(&mut r) * h as f32, (0.1 + rnd(&mut r) * 0.3) * w.min(h) as f32);
        let (pw, ph) = if i % 3 == 0 { (h, w) } else { (w, h) };
        let img = image::RgbImage::from_fn(pw, ph, |x, y| {
            let t = ((x as f32 / pw as f32 - 0.5) * ang.cos() + (y as f32 / ph as f32 - 0.5) * ang.sin() + 0.5).clamp(0.0, 1.0);
            let mut c = [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            if d < rad {
                c = c2;
            }
            let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) % 17) as f32 - 8.0;
            image::Rgb([(c[0] + n).clamp(0.0, 255.0) as u8, (c[1] + n).clamp(0.0, 255.0) as u8, (c[2] + n).clamp(0.0, 255.0) as u8])
        });
        let sub = dir.join(format!("{:03}", i / 1000));
        std::fs::create_dir_all(&sub)?;
        let bytes = mosaic_core::decode::encode_jpeg(&img, 85)?;
        std::fs::write(sub.join(format!("synthetic_{i:06}.jpg")), bytes)?;
        if i % 500 == 0 {
            job.report("generating photos", i as u64, count as u64);
        }
        Ok(())
    })?;
    println!("\nWrote {count} JPEGs ({w}×{h}) to {} in {:.1} s", dir.display(), t.elapsed().as_secs_f64());
    Ok(())
}
