//! `mosaic` command-line interface. Every workflow of the desktop app is
//! available here and uses the same engine (`mosaic-core`).

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use mosaic_core::catalog::library_ids;
use mosaic_core::export::{export_raster, plan_raster, RasterFormat};
use mosaic_core::indexer::{update_library, IndexOptions};
use mosaic_core::job::{is_cancelled, Job};
use mosaic_core::layout::{paper_by_name, PrintSettings, PAPERS};
use mosaic_core::matching::{GridSpec, MatchError, MatchSettings, Method};
use mosaic_core::pdf::{export_pdf, plan_pdf};
use mosaic_core::pool::Pool;
use mosaic_core::project::{check_sources, libraries_refs, load_target, write_manifest, write_output_stats, OutputStats, Project};
use mosaic_core::render::{render_preview, FileSource};
use mosaic_core::shape::{Outline, Shape};
use mosaic_core::units::format_bytes;
use mosaic_core::Catalog;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod bench;

#[derive(Parser)]
#[command(name = "mosaic", version, about = "Local photo mosaic generator: build a target motif from your own photo libraries.")]
struct Cli {
    /// Where the catalogue (index, descriptors, thumbnails) is stored.
    #[arg(long, global = true, env = "MOSAIC_DATA_DIR")]
    data_dir: Option<PathBuf>,
    /// Worker threads (default: number of CPU cores, max 16).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Manage photo libraries (collections of source photos).
    #[command(subcommand)]
    Library(LibCmd),
    /// Explain the mosaic methods.
    Methods,
    /// Match a target motif against libraries and save a project.
    Create(CreateArgs),
    /// Render a fast preview of a saved project from thumbnails.
    Preview { project: PathBuf, #[arg(short, long)] out: PathBuf, #[arg(long, default_value_t = 24)] tile_px: u32 },
    /// Check that every source photo of a project is present and unchanged.
    Check { project: PathBuf },
    /// Export the full mosaic as a raster image (BigTIFF, PNG or JPEG).
    Export(ExportArgs),
    /// Export a tiled, print-ready PDF.
    Pdf(PdfArgs),
    /// Write a CSV manifest of the photos used, with attribution metadata.
    Manifest { project: PathBuf, #[arg(short, long)] out: PathBuf },
    /// Suggest motifs that suit the selected libraries and settings.
    Recommend(RecommendArgs),
    /// Benchmarks with synthetic data.
    #[command(subcommand)]
    Bench(bench::BenchCmd),
}

#[derive(Subcommand)]
enum LibCmd {
    /// Create an empty library.
    Create { name: String },
    /// List libraries with their folders and counts.
    List,
    /// Remove a library from the catalogue (photos on disk are not touched).
    Remove { name: String },
    Rename { name: String, new_name: String },
    /// Add a source folder to a library.
    AddFolder { name: String, folder: PathBuf },
    /// Remove a source folder from a library (photos on disk are not touched).
    RemoveFolder { name: String, folder: PathBuf },
    /// Point a source folder at its new location without re-analysing unchanged photos.
    Relink { name: String, old: PathBuf, new: PathBuf, #[arg(long)] force: bool },
    /// Scan folders and update the index incrementally.
    Update { name: String, #[arg(long)] verify: bool },
    /// List files that were skipped and why.
    Skipped { name: String },
    /// Delete cached descriptors/thumbnails no library refers to.
    Gc,
}

#[derive(Args, Clone)]
pub struct MatchArgs {
    /// Comma-separated library names to combine.
    #[arg(long, value_delimiter = ',', required = true)]
    pub libs: Vec<String>,
    #[arg(long)]
    pub width_mm: f64,
    #[arg(long)]
    pub height_mm: f64,
    /// Approximate tile size in mm (sets the grid).
    #[arg(long, conflicts_with_all = ["cols", "rows"])]
    pub tile_mm: Option<f64>,
    #[arg(long, requires = "rows")]
    pub cols: Option<u32>,
    #[arg(long, requires = "cols")]
    pub rows: Option<u32>,
    /// classic | structure | variety
    #[arg(long, default_value = "structure")]
    pub method: String,
    /// Max uses per photo, or "unlimited".
    #[arg(long)]
    pub max_reuse: Option<String>,
    /// Extra cost (ΔE) for repeating a photo nearby.
    #[arg(long)]
    pub repeat_penalty: Option<f32>,
    #[arg(long)]
    pub repeat_radius: Option<u32>,
    /// Extra cost (ΔE) per previous use (variety method).
    #[arg(long)]
    pub usage_penalty: Option<f32>,
    /// Consider off-centre crops too.
    #[arg(long)]
    pub crop_offsets: Option<bool>,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Bounded colour correction, 0–0.5 (off by default).
    #[arg(long, default_value_t = 0.0)]
    pub color_correction: f32,
    /// Motif blend opacity, 0–0.5 (off by default).
    #[arg(long, default_value_t = 0.0)]
    pub blend: f32,
    /// Treat near-identical images (resized copies) as distinct photos.
    #[arg(long)]
    pub keep_near_duplicates: bool,
    #[arg(long)]
    pub candidates: Option<usize>,
}

#[derive(Args)]
struct CreateArgs {
    /// Target motif image.
    #[arg(long)]
    motif: PathBuf,
    #[command(flatten)]
    m: MatchArgs,
    /// Project file to write.
    #[arg(short, long)]
    out: PathBuf,
    /// Also write a preview PNG.
    #[arg(long)]
    preview: Option<PathBuf>,
    /// Outline of the mosaic: ellipse, heart, hexagon, diamond, star, a shape
    /// file (SVG; PNG with transparency; or any image where dark = inside),
    /// or "motif" to use the motif's own transparency.
    #[arg(long)]
    shape: Option<String>,
    /// Swap inside and outside of a shape file.
    #[arg(long, requires = "shape")]
    shape_invert: bool,
    /// Share of a tile's area that must lie inside the shape (0.05–1).
    #[arg(long, default_value_t = 0.5, requires = "shape")]
    shape_threshold: f32,
    /// Colour outside the shape, as #rrggbb.
    #[arg(long, default_value = "#ffffff", requires = "shape")]
    background: String,
}

fn parse_hex_color(s: &str) -> Result<[u8; 3]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("colour must look like #rrggbb, got '{s}'");
    }
    let v = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).unwrap();
    Ok([v(0), v(2), v(4)])
}

#[derive(Args)]
struct ExportArgs {
    project: PathBuf,
    #[arg(short, long)]
    out: PathBuf,
    #[arg(long, default_value_t = 300.0)]
    dpi: f64,
    /// tiff | tiff-raw | png | jpeg (default from extension).
    #[arg(long)]
    format: Option<String>,
    /// Only print the export estimate.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Args)]
struct PdfArgs {
    project: PathBuf,
    #[arg(short, long)]
    out: PathBuf,
    /// A4, A3, A2, A1, A0, Letter, Legal, Tabloid, or WIDTHxHEIGHT in mm.
    #[arg(long, default_value = "A4")]
    paper: String,
    #[arg(long)]
    landscape: bool,
    /// Printer margin in mm: one value or top,right,bottom,left.
    #[arg(long, value_delimiter = ',', default_value = "6")]
    margin: Vec<f64>,
    #[arg(long, default_value_t = 10.0)]
    overlap: f64,
    #[arg(long, default_value_t = 300.0)]
    dpi: f64,
    #[arg(long)]
    no_marks: bool,
    #[arg(long, default_value_t = 92)]
    quality: u8,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Args)]
struct RecommendArgs {
    #[command(flatten)]
    m: MatchArgs,
    /// Folder of candidate motif images (independent of the libraries).
    #[arg(long)]
    folder: Option<PathBuf>,
    /// Also evaluate up to N photos from the selected libraries as motifs.
    #[arg(long, default_value_t = 0)]
    from_libraries: usize,
    /// Number of candidates that get a low-resolution trial mosaic.
    #[arg(long, default_value_t = 8)]
    trials: usize,
    /// Folder for trial previews.
    #[arg(long)]
    previews: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    top: usize,
}

fn main() {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        let _ = rayon_init(t);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let c = cancel.clone();
        let _ = ctrlc::set_handler(move || {
            if c.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            eprintln!("\ncancelling… (press Ctrl-C again to abort immediately)");
        });
    }
    match run(cli, cancel) {
        Ok(()) => {}
        Err(e) if is_cancelled(&e) => {
            eprintln!("\ncancelled; no output was written");
            std::process::exit(130);
        }
        Err(e) => {
            if let Some(MatchError::Infeasible(inf)) = e.downcast_ref::<MatchError>() {
                eprintln!("\nCannot build this mosaic: {}", inf.message);
                eprintln!("Options:");
                for s in &inf.suggestions {
                    eprintln!("  - {s}");
                }
                std::process::exit(2);
            }
            eprintln!("\nerror: {e:#}");
            std::process::exit(1);
        }
    }
}

fn rayon_init(threads: usize) -> Result<()> {
    mosaic_core::init_threads(threads)
}

pub fn data_dir(cli: &Option<PathBuf>) -> PathBuf {
    cli.clone().unwrap_or_else(|| {
        directories::ProjectDirs::from("", "", "mosaic").map(|d| d.data_dir().to_path_buf()).unwrap_or_else(|| PathBuf::from(".mosaic"))
    })
}

pub fn progress_job(cancel: Arc<AtomicBool>) -> Job {
    let last = Mutex::new((Instant::now(), String::new()));
    Job::new(move |p| {
        let mut l = last.lock().unwrap();
        if l.0.elapsed().as_millis() < 200 && l.1 == p.stage && p.done != p.total {
            return;
        }
        *l = (Instant::now(), p.stage.clone());
        if p.total > 0 {
            eprint!("\r\x1b[2K{}: {}/{} ({:.0}%)", p.stage, p.done, p.total, p.done as f64 * 100.0 / p.total as f64);
        } else {
            eprint!("\r\x1b[2K{}…", p.stage);
        }
        let _ = std::io::stderr().flush();
    })
    .with_cancel_flag(cancel)
}

pub fn settings_from(m: &MatchArgs) -> Result<(GridSpec, MatchSettings)> {
    let grid = match (m.tile_mm, m.cols, m.rows) {
        (Some(t), _, _) => GridSpec::from_tile_mm(m.width_mm, m.height_mm, t)?,
        (None, Some(c), Some(r)) => GridSpec::new(m.width_mm, m.height_mm, c, r)?,
        _ => bail!("give --tile-mm or --cols and --rows"),
    };
    let mut s = MatchSettings::for_method(Method::parse(&m.method)?);
    if let Some(r) = &m.max_reuse {
        s.max_reuse = if r.eq_ignore_ascii_case("unlimited") { None } else { Some(r.parse().context("--max-reuse must be a number or 'unlimited'")?) };
    }
    if let Some(v) = m.repeat_penalty {
        s.repeat_penalty = v;
    }
    if let Some(v) = m.repeat_radius {
        s.repeat_radius = v;
    }
    if let Some(v) = m.usage_penalty {
        s.usage_penalty = v;
    }
    if let Some(v) = m.crop_offsets {
        s.crop_offsets = v;
    }
    if let Some(v) = m.candidates {
        s.candidates = v;
    }
    s.seed = m.seed;
    s.color_correction = m.color_correction;
    s.motif_blend = m.blend;
    s.merge_near_duplicates = !m.keep_near_duplicates;
    s.validate()?;
    Ok((grid, s))
}

fn run(cli: Cli, cancel: Arc<AtomicBool>) -> Result<()> {
    let dd = data_dir(&cli.data_dir);
    let threads = cli.threads.unwrap_or_else(mosaic_core::indexer::default_threads);
    let job = progress_job(cancel);
    match cli.cmd {
        Cmd::Methods => {
            for m in Method::ALL {
                println!("{} ({})\n  {}\n", m.title(), m.name(), m.explanation());
            }
            println!("Reuse limit (--max-reuse) is a hard limit per photo across all selected libraries; duplicates count as one photo.");
            println!("Colour correction and motif blend are off by default; when enabled they alter the source photos' appearance and are recorded in the project.");
        }
        Cmd::Library(c) => library(&dd, c, threads, &job)?,
        Cmd::Create(a) => {
            let cat = Catalog::open(&dd)?;
            let (grid, settings) = settings_from(&a.m)?;
            let libs = library_ids(&cat, &a.m.libs)?;
            let t = Instant::now();
            let pool = Pool::load(&cat, &libs, settings.merge_near_duplicates)?;
            let load_ms = t.elapsed().as_millis();
            let (target, motif) = load_target(&a.motif, &grid)?;
            let outline = match &a.shape {
                Some(name) => {
                    let shape = if name.eq_ignore_ascii_case("motif") {
                        // The motif's own transparency, aligned with the picture.
                        let mut s = Shape::from_file(&a.motif, a.shape_invert)?;
                        if let Shape::File { cover, .. } = &mut s {
                            *cover = true;
                        }
                        s
                    } else { Shape::parse(name, a.shape_invert)? };
                    let o = Outline { shape, background: parse_hex_color(&a.background)?, threshold: a.shape_threshold };
                    o.validate()?;
                    Some(o)
                }
                None => None,
            };
            let project = Project::create_with_outline(&pool, libraries_refs(&cat, &libs)?, &target, motif, grid, settings, outline, &job).map_err(anyhow::Error::from)?;
            eprintln!();
            project.save(&a.out)?;
            let s = &project.stats;
            println!("Saved project {}", a.out.display());
            println!(
                "Grid {}×{} = {} tiles, each {:.1}×{:.1} mm. Method: {}.",
                grid.cols, grid.rows, grid.tiles(), grid.tile_mm().0, grid.tile_mm().1, project.settings.method.title()
            );
            if let Some(o) = &project.outline {
                println!("Shape: {} – {} of {} tiles are inside and get a photo.", o.shape.label(), project.placed_tiles(), grid.tiles());
            }
            println!(
                "Pool: {} distinct photos ({} groups after duplicate merging), {} candidates (crops).",
                s.pool_photos, s.pool_groups, s.candidates
            );
            println!(
                "Used {} distinct photos, max {} uses of one photo, {} adjacent repeats. Mean colour error ΔE {:.1}.",
                s.distinct_used, s.max_uses, s.nearby_repeats, s.mean_delta_e
            );
            println!(
                "Timings: load pool {} ms, candidates+index {:.0} ms, retrieval {:.0} ms, assignment {:.0} ms.",
                load_ms, s.build_ms, s.retrieval_ms, s.assign_ms
            );
            if s.widened_searches > 0 {
                println!("Search widened for {} tiles (reuse limits exhausted nearest candidates); available-photo index rebuilds: {}.", s.widened_searches, s.index_rebuilds);
            }
            if project.settings.color_correction > 0.0 || project.settings.motif_blend > 0.0 {
                println!(
                    "Disclosure: colour correction {:.0}% and motif blend {:.0}% alter how source photos look.",
                    project.settings.color_correction * 100.0,
                    project.settings.motif_blend * 100.0
                );
            }
            if let Some(p) = a.preview {
                let t = Instant::now();
                let img = render_preview(&cat, &project, 24, &job)?;
                save_png(&img, &p)?;
                eprintln!();
                println!("Preview {} ({}×{} px) in {} ms", p.display(), img.width(), img.height(), t.elapsed().as_millis());
            }
        }
        Cmd::Preview { project, out, tile_px } => {
            let cat = Catalog::open(&dd)?;
            let p = Project::load(&project)?;
            let t = Instant::now();
            let img = render_preview(&cat, &p, tile_px, &job)?;
            save_png(&img, &out)?;
            eprintln!();
            println!("Preview {} ({}×{} px) in {} ms", out.display(), img.width(), img.height(), t.elapsed().as_millis());
        }
        Cmd::Check { project } => {
            let cat = Catalog::open(&dd)?;
            let p = Project::load(&project)?;
            let chk = check_sources(&cat, &p, &job)?;
            eprintln!();
            println!("{}", chk.report(&p));
            if !chk.ok() {
                std::process::exit(3);
            }
        }
        Cmd::Export(a) => {
            let cat = Catalog::open(&dd)?;
            let p = Project::load(&a.project)?;
            let fmt = match &a.format {
                Some(f) => RasterFormat::parse(f)?,
                None => RasterFormat::from_path(&a.out).unwrap_or(RasterFormat::Tiff),
            };
            let plan = plan_raster(&p, a.dpi, fmt)?;
            print!("{}", plan.summary());
            if a.dry_run {
                return Ok(());
            }
            let src = resolve_files(&cat, &p, &job)?;
            let rep = export_raster(&p, &src, &plan, &a.out, &job)?;
            eprintln!();
            println!(
                "Exported {} ({}) in {:.1} s. Tile cache: {} hits, {} renders.",
                rep.path,
                format_bytes(rep.bytes),
                rep.elapsed_ms as f64 / 1000.0,
                rep.cache_hits,
                rep.cache_misses
            );
            let details = serde_json::json!({ "format": format!("{:?}", plan.format), "dpi": plan.spec.dpi, "width_px": plan.spec.width_px, "height_px": plan.spec.height_px, "bytes": rep.bytes, "seconds": rep.elapsed_ms as f64 / 1000.0 });
            print_stats(write_output_stats(&cat, &p, &a.out, "raster", details)?);
        }
        Cmd::Pdf(a) => {
            let cat = Catalog::open(&dd)?;
            let p = Project::load(&a.project)?;
            let mut ps = match paper_by_name(&a.paper) {
                Some(pp) => PrintSettings::new(pp),
                None => {
                    let (w, h) = a
                        .paper
                        .to_ascii_lowercase()
                        .split_once('x')
                        .and_then(|(w, h)| Some((w.trim().parse::<f64>().ok()?, h.trim().parse::<f64>().ok()?)))
                        .with_context(|| format!("unknown paper '{}'; use {} or WIDTHxHEIGHT in mm", a.paper, PAPERS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
                    PrintSettings::custom(w, h)
                }
            };
            ps.landscape = a.landscape;
            ps.margins_mm = match a.margin.as_slice() {
                [m] => [*m; 4],
                [t, r, b, l] => [*t, *r, *b, *l],
                _ => bail!("--margin takes one value or four (top,right,bottom,left)"),
            };
            ps.overlap_mm = a.overlap;
            ps.dpi = a.dpi;
            ps.marks = !a.no_marks;
            ps.jpeg_quality = a.quality;
            let plan = plan_pdf(&p, &ps)?;
            print!("{}", plan.summary());
            if a.dry_run {
                return Ok(());
            }
            let src = resolve_files(&cat, &p, &job)?;
            let overview = render_preview(&cat, &p, overview_tile_px(&p), &job)?;
            let rep = export_pdf(&p, &src, Some(&overview), &plan, &a.out, &job)?;
            eprintln!();
            println!("Exported {} ({}, {} pages) in {:.1} s.", rep.path, format_bytes(rep.bytes), plan.printed_pages() + 1, rep.elapsed_ms as f64 / 1000.0);
            let (sw, sh) = plan.print.sheet_mm();
            let details = serde_json::json!({ "paper": plan.print.paper_name, "sheet_mm": [sw, sh], "landscape": plan.print.landscape, "pages": plan.printed_pages(), "blank_pages_skipped": plan.blank_pages(), "page_grid": [plan.layout.cols, plan.layout.rows], "overlap_mm": plan.print.overlap_mm, "margins_mm": plan.print.margins_mm, "dpi": plan.print.dpi, "bytes": rep.bytes });
            print_stats(write_output_stats(&cat, &p, &a.out, "tiled-pdf", details)?);
        }
        Cmd::Manifest { project, out } => {
            let cat = Catalog::open(&dd)?;
            let p = Project::load(&project)?;
            let n = write_manifest(&cat, &p, &out)?;
            println!("Wrote {} ({n} photos). Usage rights are not inferred; check each photo's licence.", out.display());
        }
        Cmd::Recommend(a) => recommend_cmd(&dd, a, &job)?,
        Cmd::Bench(b) => bench::run(&dd, b, &job)?,
    }
    Ok(())
}

fn print_stats((path, st): (PathBuf, OutputStats)) {
    let x = &st.exif;
    println!(
        "Statistics: {} tiles, {} distinct photos ({} used once, max {} uses); EXIF on {} photos, none on {}.",
        st.tiles, st.distinct_photos_used, st.photos_used_once, st.max_uses_of_one_photo, x.photos_with_exif, x.photos_without_exif
    );
    let r = |name: &str, o: &Option<mosaic_core::project::Range>, f: &dyn Fn(f64) -> String| {
        if let Some(r) = o {
            println!("  {name}: {} – {} (median {})", f(r.min), f(r.max), f(r.median));
        }
    };
    r("aperture", &x.aperture_f, &|v| format!("f/{v:.1}"));
    r("shutter", &x.shutter_s, &|v| mosaic_core::project::shutter_label(v));
    r("ISO", &x.iso, &|v| format!("{v:.0}"));
    r("focal length", &x.focal_length_mm, &|v| format!("{v:.0} mm"));
    if let Some(c) = x.cameras.first() {
        println!("  most used camera: {} ({} photos)", c.value, c.photos);
    }
    println!("  saved {}", path.display());
}

pub fn overview_tile_px(p: &Project) -> u32 {
    (1600 / p.grid.cols.max(p.grid.rows)).clamp(2, 32)
}

fn resolve_files(cat: &Catalog, p: &Project, job: &Job) -> Result<FileSource> {
    let chk = check_sources(cat, p, job)?;
    if !chk.ok() {
        bail!("{}", chk.report(p));
    }
    Ok(FileSource { paths: chk.resolved.into_iter().map(|p| p.unwrap()).collect() })
}

pub fn save_png(img: &image::RgbImage, out: &Path) -> Result<()> {
    let mut t = mosaic_core::atomic::TempOutput::create(out)?;
    img.write_to(&mut std::io::BufWriter::new(t.file()), image::ImageFormat::Png)?;
    t.commit()
}

fn library(dd: &Path, c: LibCmd, threads: usize, job: &Job) -> Result<()> {
    let cat = Catalog::open(dd)?;
    match c {
        LibCmd::Create { name } => {
            cat.create_library(&name)?;
            println!("Created library '{name}'");
        }
        LibCmd::List => {
            let libs = cat.libraries()?;
            if libs.is_empty() {
                println!("No libraries. Create one with: mosaic library create NAME");
            }
            for l in libs {
                println!("{}: {} distinct photos, {} files indexed, {} skipped", l.name, l.distinct_photos, l.files_ok, l.files_skipped);
                for r in l.roots {
                    let state = if r.path.is_dir() { "" } else { "  [MISSING — relink or reconnect]" };
                    println!("    {}{state}", r.path.display());
                }
            }
        }
        LibCmd::Remove { name } => {
            cat.remove_library(cat.library_id(&name)?)?;
            println!("Removed library '{name}' from the catalogue. No photos were deleted.");
        }
        LibCmd::Rename { name, new_name } => cat.rename_library(cat.library_id(&name)?, &new_name)?,
        LibCmd::AddFolder { name, folder } => {
            cat.add_root(cat.library_id(&name)?, &folder)?;
            println!("Added {} to '{name}'. Run: mosaic library update {name}", folder.display());
        }
        LibCmd::RemoveFolder { name, folder } => {
            cat.remove_root(cat.library_id(&name)?, &folder)?;
            println!("Removed folder from '{name}'. No photos were deleted.");
        }
        LibCmd::Relink { name, old, new, force } => {
            let r = cat.relink_root(cat.library_id(&name)?, &old, &new, force)?;
            println!("Relinked. {}/{} sampled files found at the new location. Run 'library update' to pick up any changes.", r.found, r.sampled);
        }
        LibCmd::Update { name, verify } => {
            let id = cat.library_id(&name)?;
            let opts = IndexOptions { threads, verify, ..Default::default() };
            let r = update_library(&cat, id, &opts, job)?;
            eprintln!();
            println!(
                "{}: {} files seen, {} added, {} changed, {} removed, {} unchanged.",
                name, r.files_seen, r.added, r.changed, r.removed, r.unchanged
            );
            println!(
                "Decoded {} new photos, reused {} cached descriptors, skipped {}. {} duplicate files. {} non-image files ignored.",
                r.decoded, r.reused_descriptors, r.skipped.len(), r.duplicate_files, r.ignored_other_files
            );
            let secs = r.elapsed_ms as f64 / 1000.0;
            println!(
                "Time {:.2} s (folder scan {:.2} s){}",
                secs,
                r.walk_ms as f64 / 1000.0,
                if r.decoded > 0 { format!(", {:.0} photos/s decoded", r.decoded as f64 / (secs - r.walk_ms as f64 / 1000.0).max(1e-3)) } else { String::new() }
            );
            for (p, why) in r.skipped.iter().take(10) {
                println!("  skipped {}: {why}", p.display());
            }
            if r.skipped.len() > 10 {
                println!("  … see 'mosaic library skipped {name}'");
            }
            for m in &r.roots_missing {
                println!("  folder missing (records kept): {} — reconnect it or use 'library relink'", m.display());
            }
            for e in r.walk_errors.iter().take(5) {
                println!("  scan error: {e}");
            }
            if r.cancelled {
                println!("Cancelled: photos processed so far are saved; run update again to continue.");
            }
        }
        LibCmd::Skipped { name } => {
            for (p, why) in cat.skipped(cat.library_id(&name)?)? {
                println!("{}\t{why}", p.display());
            }
        }
        LibCmd::Gc => println!("Removed {} unused descriptors", cat.gc()?),
    }
    Ok(())
}

fn recommend_cmd(dd: &Path, a: RecommendArgs, job: &Job) -> Result<()> {
    use mosaic_core::recommend::{recommend, MotifCandidate, RecommendOptions};
    let cat = Catalog::open(dd)?;
    let (grid, settings) = settings_from(&a.m)?;
    let libs = library_ids(&cat, &a.m.libs)?;
    let pool = Pool::load(&cat, &libs, settings.merge_near_duplicates)?;
    let mut cands = Vec::new();
    if let Some(f) = &a.folder {
        cands.extend(MotifCandidate::from_folder(f)?);
    }
    if a.from_libraries > 0 {
        cands.extend(MotifCandidate::from_pool(&pool, a.from_libraries, settings.seed));
    }
    if cands.is_empty() {
        bail!("give --folder and/or --from-libraries N");
    }
    let opts = RecommendOptions { trials: a.trials, preview_dir: a.previews.clone(), ..Default::default() };
    let t = Instant::now();
    let recs = recommend(&cat, &pool, cands, &grid, &settings, &opts, job)?;
    eprintln!();
    println!(
        "Evaluated {} motifs for {} photos, {}×{} grid, {} method, reuse {} in {:.1} s.",
        recs.len(),
        pool.photos.len(),
        grid.cols,
        grid.rows,
        settings.method.name(),
        settings.max_reuse.map(|m| m.to_string()).unwrap_or("unlimited".into()),
        t.elapsed().as_secs_f64()
    );
    println!("Scores estimate technical suitability for this collection, not artistic quality.\n");
    for (i, r) in recs.iter().take(a.top).enumerate() {
        println!("{:>2}. {:.0}/100  {}", i + 1, r.score * 100.0, r.label);
        println!("    {}", r.notes.join("; "));
        if let Some(p) = &r.preview {
            println!("    trial preview: {}", p.display());
        }
    }
    Ok(())
}
