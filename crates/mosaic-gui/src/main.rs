//! Desktop interface (Slint). All work is done by `mosaic-core` on worker
//! threads; the UI thread only reads settings and shows results.
//!
//! The window has five colour-coded steps (Photos, Picture, Size & style,
//! Export, Statistics) and a preview that is always visible. Photo libraries are handled
//! automatically: folders go into one "My photos" collection, are scanned when
//! added and re-checked on every start. The last mosaic is autosaved and
//! reopened on start.

#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{anyhow, bail, Context, Result};
use mosaic_core::catalog::{Catalog, LibraryInfo};
use mosaic_core::export::{export_raster, plan_raster, RasterFormat};
use mosaic_core::decode::{decode_file, resize_crop};
use mosaic_core::indexer::{update_library, walk_images, IndexOptions, IMAGE_EXTENSIONS};
use mosaic_core::job::{is_cancelled, Job};
use mosaic_core::layout::{PrintSettings, PAPERS};
use mosaic_core::matching::{GridSpec, MatchError, MatchSettings, Method};
use mosaic_core::pdf::{export_pdf, plan_pdf};
use mosaic_core::pool::Pool;
use mosaic_core::project::{check_sources, libraries_refs, load_target, output_stats, shutter_label, write_manifest, write_output_stats, Counted, OutputStats, Range, Project};
use mosaic_core::recommend::{recommend, MotifCandidate, MotifSource, RecommendOptions};
use mosaic_core::render::{render_preview, FileSource};
use mosaic_core::shape::{Outline, Shape};
use mosaic_core::units::format_bytes;
use slint::{ComponentHandle, Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, VecModel};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

slint::include_modules!();

/// Name of the collection that folders are added to when the user has not
/// set up named libraries with the CLI.
const DEFAULT_COLLECTION: &str = "My photos";
/// Autosaved copy of the last mosaic, reopened on start.
const LAST_PROJECT: &str = "last-project.json";
/// Style presets: Sharp, Balanced, Varied → method and reuse limit.
const STYLES: [(Method, u32, &str); 3] = [
    (Method::Classic, 5, "Sharp: matches the colour of each tile as closely as possible. Fastest. Each photo is used at most 5 times."),
    (Method::Structure, 3, "Balanced: also follows edges and shading inside each tile, so outlines look cleaner. Each photo is used at most 3 times."),
    (Method::Variety, 2, "Varied: spreads the work over as many different photos as possible and avoids repeats nearby. Each photo is used at most 2 times."),
];

#[derive(Default)]
struct State {
    data_dir: PathBuf,
    libs: Vec<LibraryInfo>,
    /// Collections the user unticked. Everything else is used.
    excluded: HashSet<i64>,
    /// (library id, folder path) for each row of the folder list.
    folders: Vec<(i64, PathBuf)>,
    project: Option<Arc<Project>>,
    pool: Option<(Vec<i64>, bool, Arc<Pool>)>,
    job: Option<Job>,
    suggestion_paths: Vec<Option<PathBuf>>,
    /// Statistics of the current project, once counted.
    stats: Option<Arc<OutputStats>>,
    /// Folder being checked before it is added: (check number, path).
    check: Option<(u64, PathBuf)>,
    check_counter: u64,
}

type Shared = Arc<Mutex<State>>;
type Apply = Box<dyn FnOnce(&AppWindow, &mut State) + Send>;

fn data_dir() -> PathBuf {
    std::env::var_os("MOSAIC_DATA_DIR").map(PathBuf::from).unwrap_or_else(|| {
        directories::ProjectDirs::from("", "", "mosaic").map(|d| d.data_dir().to_path_buf()).unwrap_or_else(|| PathBuf::from(".mosaic"))
    })
}

fn to_image(img: &image::RgbImage) -> SharedPixelBuffer<Rgb8Pixel> {
    SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(img.as_raw(), img.width(), img.height())
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Run `work` on a worker thread with progress, cancellation and error display.
fn run_job(ui: &AppWindow, st: &Shared, label: &str, work: impl FnOnce(&Job, PathBuf) -> Result<Apply> + Send + 'static) {
    let weak = ui.as_weak();
    let last = Arc::new(Mutex::new(Instant::now()));
    let w2 = weak.clone();
    let job = Job::new(move |p| {
        let mut l = last.lock().unwrap();
        if l.elapsed().as_millis() < 100 && p.done != p.total {
            return;
        }
        *l = Instant::now();
        let (stage, done, total) = (p.stage.clone(), p.done, p.total);
        let _ = w2.upgrade_in_event_loop(move |ui| {
            ui.set_progress_known(total > 0);
            ui.set_progress(if total > 0 { done as f32 / total as f32 } else { 0.0 });
            ui.set_status(if total > 0 { format!("{stage}: {done} / {total}") } else { format!("{stage}…") }.into());
        });
    });
    st.lock().unwrap().job = Some(job.clone());
    ui.set_busy(true);
    ui.set_notice("".into());
    ui.set_status(format!("{label}…").into());
    let dd = st.lock().unwrap().data_dir.clone();
    let st2 = st.clone();
    let label = label.to_string();
    std::thread::spawn(move || {
        let t = Instant::now();
        let r = work(&job, dd);
        let secs = t.elapsed().as_secs_f64();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            let mut s = st2.lock().unwrap();
            s.job = None;
            ui.set_busy(false);
            ui.set_progress_known(false);
            match r {
                Ok(apply) => {
                    ui.set_status(format!("{label}: done in {secs:.1} s").into());
                    apply(&ui, &mut s);
                }
                Err(e) if is_cancelled(&e) => ui.set_status(format!("{label}: cancelled. Nothing was written.").into()),
                Err(e) => {
                    let msg = match e.downcast_ref::<MatchError>() {
                        Some(MatchError::Infeasible(inf)) => format!("{}\nWhat you can do:\n  • {}", inf.message, inf.suggestions.join("\n  • ")),
                        _ => format!("{e:#}"),
                    };
                    ui.set_status(format!("{label}: failed").into());
                    ui.set_notice(format!("{label} failed. {msg}").into());
                }
            }
        });
    });
}

fn selected_ids(st: &State) -> Vec<i64> {
    let mut v: Vec<i64> = st.libs.iter().filter(|l| !st.excluded.contains(&l.id)).map(|l| l.id).collect();
    v.sort_unstable();
    v
}

/// Shows the check mark in the status bar when every folder was found and
/// scanned and at least one photo is indexed. Call after a scan finishes.
fn mark_indexed(ui: &AppWindow, st: &State, scan_ok: bool) {
    ui.set_photos_indexed(scan_ok && ui.get_photo_count() > 0 && ui_missing(st) == 0);
}

fn refresh_libraries(ui: &AppWindow, st: &mut State) -> Result<()> {
    let cat = Catalog::open(&st.data_dir)?;
    st.libs = cat.libraries()?;
    let ids: HashSet<i64> = st.libs.iter().map(|l| l.id).collect();
    st.excluded.retain(|i| ids.contains(i));

    let named = st.libs.len() > 1;
    st.folders.clear();
    let mut rows = Vec::new();
    for l in &st.libs {
        for r in &l.roots {
            st.folders.push((l.id, r.path.clone()));
            rows.push(FolderRow {
                path: if named { format!("{}  ({})", r.path.display(), l.name) } else { r.path.display().to_string() }.into(),
                collection: l.name.clone().into(),
                missing: !r.path.is_dir(),
            });
        }
    }
    ui.set_folders(ModelRc::new(VecModel::from(rows)));

    let cols: Vec<CollectionRow> = st
        .libs
        .iter()
        .map(|l| CollectionRow { name: l.name.clone().into(), summary: format!("{} photos", thousands(l.distinct_photos)).into(), selected: !st.excluded.contains(&l.id) })
        .collect();
    ui.set_collections(ModelRc::new(VecModel::from(cols)));

    let used: Vec<&LibraryInfo> = st.libs.iter().filter(|l| !st.excluded.contains(&l.id)).collect();
    let photos: u64 = used.iter().map(|l| l.distinct_photos).sum();
    let skipped: u64 = used.iter().map(|l| l.files_skipped).sum();
    let missing = ui_missing(st);
    ui.set_photo_count(photos.min(i32::MAX as u64) as i32);
    ui.set_skipped_count(skipped.min(i32::MAX as u64) as i32);
    ui.set_photos_summary(
        match (photos, missing) {
            (0, 0) => "No photos yet".to_string(),
            (n, 0) => format!("{} photos", thousands(n)),
            (n, m) => format!("{} photos · {m} folder(s) missing", thousands(n)),
        }
        .into(),
    );
    Ok(())
}

fn ui_missing(st: &State) -> usize {
    st.folders.iter().filter(|(_, p)| !p.is_dir()).count()
}

/// The collection new folders go into: the only one there is, or "My photos".
fn default_library(cat: &Catalog, libs: &[LibraryInfo]) -> Result<i64> {
    if let Ok(id) = cat.library_id(DEFAULT_COLLECTION) {
        return Ok(id);
    }
    if libs.len() == 1 {
        return Ok(libs[0].id);
    }
    cat.create_library(DEFAULT_COLLECTION)
}

fn num(s: SharedString, what: &str) -> Result<f64> {
    s.trim().replace(',', ".").parse::<f64>().with_context(|| format!("{what} must be a number"))
}

fn read_settings(ui: &AppWindow) -> Result<(GridSpec, MatchSettings)> {
    let (w, h) = (num(ui.get_width_mm(), "Width")?, num(ui.get_height_mm(), "Height")?);
    let grid = if ui.get_use_grid() {
        GridSpec::new(w, h, ui.get_cols().max(1) as u32, ui.get_rows().max(1) as u32)?
    } else {
        GridSpec::from_tile_mm(w, h, num(ui.get_tile_mm(), "Tile size")?)?
    };
    let method = STYLES[ui.get_style_index().clamp(0, 2) as usize].0;
    let mut s = MatchSettings::for_method(method);
    s.max_reuse = if ui.get_unlimited_reuse() { None } else { Some(ui.get_max_reuse().max(1) as u32) };
    s.repeat_penalty = ui.get_repeat_penalty();
    s.repeat_radius = ui.get_repeat_radius().max(1) as u32;
    s.usage_penalty = if method == Method::Variety { ui.get_usage_penalty() } else { 0.0 };
    s.crop_offsets = ui.get_crop_offsets();
    s.seed = ui.get_seed().max(0) as u64;
    s.color_correction = ui.get_color_correction();
    s.motif_blend = ui.get_motif_blend();
    s.validate()?;
    Ok((grid, s))
}

const SHAPES: [Shape; 5] = [Shape::Ellipse, Shape::Heart, Shape::Hexagon, Shape::Diamond, Shape::Star];
const BACKGROUNDS: [[u8; 3]; 3] = [[255, 255, 255], [0, 0, 0], [220, 220, 220]];

fn read_outline(ui: &AppWindow) -> Result<Option<Outline>> {
    let i = ui.get_shape_index();
    let shape = match i {
        1..=5 => SHAPES[i as usize - 1].clone(),
        6 => {
            let f = ui.get_shape_file();
            if f.is_empty() {
                bail!("choose a shape file, or pick another shape");
            }
            Shape::from_file(Path::new(f.as_str()), ui.get_shape_invert())?
        }
        _ => return Ok(None),
    };
    let bg = BACKGROUNDS[ui.get_shape_background_index().clamp(0, 2) as usize];
    let o = Outline { shape, background: bg, threshold: ui.get_shape_threshold().clamp(0.05, 1.0) };
    o.validate()?;
    Ok(Some(o))
}

fn show_outline(ui: &AppWindow, o: Option<&Outline>) {
    let Some(o) = o else {
        ui.set_shape_index(0);
        return;
    };
    match &o.shape {
        Shape::File { path, invert, .. } => {
            ui.set_shape_index(6);
            ui.set_shape_file(path.display().to_string().into());
            ui.set_shape_file_name(file_name(path).into());
            ui.set_shape_invert(*invert);
        }
        s => ui.set_shape_index(SHAPES.iter().position(|x| x == s).map(|p| p as i32 + 1).unwrap_or(0)),
    }
    ui.set_shape_background_index(BACKGROUNDS.iter().position(|b| *b == o.background).unwrap_or(0) as i32);
    ui.set_shape_threshold(o.threshold);
}

/// Applies a style preset: method defaults plus the preset's reuse limit.
fn apply_style(ui: &AppWindow, i: i32) {
    let (m, reuse, help) = STYLES[i.clamp(0, 2) as usize];
    let s = MatchSettings::for_method(m);
    ui.set_style_help(help.into());
    ui.set_repeat_penalty(s.repeat_penalty);
    ui.set_repeat_radius(s.repeat_radius as i32);
    ui.set_usage_penalty(s.usage_penalty);
    ui.set_crop_offsets(s.crop_offsets);
    ui.set_unlimited_reuse(false);
    ui.set_max_reuse(reuse as i32);
}

/// Rough viewing distance at which tiles blend into a picture (README guide:
/// 10–15 mm tiles read from about 2 m, 30 mm tiles from 4–5 m).
fn viewing_distance_m(tile_mm: f64) -> f64 {
    (tile_mm * 0.15).max(1.0)
}

fn update_grid_info(ui: &AppWindow, st: &State) {
    let style = ["Sharp", "Balanced", "Varied"][ui.get_style_index().clamp(0, 2) as usize];
    match read_settings(ui) {
        Ok((g, s)) => {
            let photos: u64 = st.libs.iter().filter(|l| !st.excluded.contains(&l.id)).map(|l| l.distinct_photos).sum();
            let (tw, th) = g.tile_mm();
            let mut t = format!("{} × {} = {} tiles of {:.0} × {:.0} mm. Looks best from about {:.0} m away.", g.cols, g.rows, thousands(g.tiles() as u64), tw, th, viewing_distance_m(tw.max(th)));
            let mut placed = g.tiles() as u64;
            match read_outline(ui).and_then(|o| o.map(|o| o.active_tiles(&g)).transpose()) {
                Ok(Some(a)) => {
                    placed = a.iter().filter(|&&x| x).count() as u64;
                    t += &format!(" {} tiles are inside the shape.", thousands(placed));
                }
                Ok(None) => {}
                Err(e) => t += &format!(" Shape: {e:#}."),
            }
            if photos > 0 {
                if let Some(m) = s.max_reuse {
                    let need = placed.div_ceil(m as u64);
                    if photos < need {
                        t += &format!(" Not enough photos: you need at least {} for this size, or use bigger tiles.", thousands(need));
                    }
                }
            }
            ui.set_grid_info(t.into());
            ui.set_size_summary(format!("{:.0} × {:.0} cm · {style}", g.width_mm / 10.0, g.height_mm / 10.0).into());
        }
        Err(e) => {
            ui.set_grid_info(format!("{e}").into());
            ui.set_size_summary(style.into());
        }
    }
}

fn load_pool(st: &Shared, cat: &Catalog, libs: &[i64], merge: bool) -> Result<Arc<Pool>> {
    if let Some((l, m, p)) = &st.lock().unwrap().pool {
        if l == libs && *m == merge {
            return Ok(p.clone());
        }
    }
    let p = Arc::new(Pool::load(cat, libs, merge)?);
    st.lock().unwrap().pool = Some((libs.to_vec(), merge, p.clone()));
    Ok(p)
}

fn selected_libs(st: &State) -> Result<Vec<i64>> {
    let v = selected_ids(st);
    let photos: u64 = st.libs.iter().filter(|l| v.contains(&l.id)).map(|l| l.distinct_photos).sum();
    if v.is_empty() || photos == 0 {
        bail!("add a folder of photos in step 1 first");
    }
    Ok(v)
}

fn print_settings(ui: &AppWindow) -> Result<PrintSettings> {
    let i = ui.get_paper_index().max(0) as usize;
    let mut ps = if i < PAPERS.len() { PrintSettings::new(PAPERS[i]) } else { PrintSettings::custom(num(ui.get_custom_w(), "Paper width")?, num(ui.get_custom_h(), "Paper height")?) };
    ps.landscape = ui.get_landscape();
    ps.margins_mm = [num(ui.get_margin_mm(), "Margin")?; 4];
    ps.overlap_mm = num(ui.get_overlap_mm(), "Overlap")?;
    ps.dpi = num(ui.get_pdf_dpi(), "DPI")?;
    ps.marks = ui.get_marks();
    Ok(ps)
}

fn raster_format(ui: &AppWindow) -> RasterFormat {
    [RasterFormat::Tiff, RasterFormat::Png, RasterFormat::Jpeg][ui.get_raster_format_index().clamp(0, 2) as usize]
}

fn report_error(ui: &AppWindow, e: anyhow::Error) {
    let msg = format!("{e:#}");
    let mut c = msg.chars();
    let msg = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
    ui.set_notice(msg.clone().into());
    ui.set_status(msg.lines().next().unwrap_or("").to_string().into());
}

fn resolve(cat: &Catalog, p: &Project, job: &Job) -> Result<FileSource> {
    let chk = check_sources(cat, p, job)?;
    if !chk.ok() {
        bail!("{}", chk.report(p));
    }
    Ok(FileSource { paths: chk.resolved.into_iter().map(|p| p.unwrap()).collect() })
}

fn show_motif(ui: &AppWindow, path: &Path) {
    match mosaic_core::decode::decode_file(path, Some(512)) {
        Ok(d) => {
            ui.set_motif_image(Image::from_rgb8(to_image(&d.image)));
            ui.set_motif_path(path.display().to_string().into());
            ui.set_motif_name(file_name(path).into());
        }
        Err(e) => report_error(ui, anyhow!("cannot open this picture: {e}")),
    }
}

fn mark_stale(ui: &AppWindow) {
    ui.set_preview_stale(ui.get_has_preview());
}

/// Short, automatic size estimate for the Export step.
fn update_export_summary(ui: &AppWindow, st: &State) {
    let Some(p) = st.project.clone() else {
        ui.set_export_summary("".into());
        return;
    };
    let text = if ui.get_export_kind() == 0 {
        num(ui.get_export_dpi(), "DPI").and_then(|dpi| plan_raster(&p, dpi, raster_format(ui))).map(|pl| {
            format!(
                "{} × {} px at {} DPI\nFile size about {}, needs about {} of memory",
                thousands(pl.spec.width_px),
                thousands(pl.spec.height_px),
                pl.spec.dpi,
                format_bytes(pl.est_output_bytes),
                format_bytes(pl.est_peak_memory)
            )
        })
    } else {
        print_settings(ui).and_then(|ps| plan_pdf(&p, &ps)).map(|pl| {
            format!(
                "{} pages ({} across × {} down{}) on {} {}, plus 1 assembly page\nFile size about {}",
                pl.printed_pages(),
                pl.layout.cols,
                pl.layout.rows,
                if pl.blank_pages() > 0 { format!(", {} blank skipped", pl.blank_pages()) } else { String::new() },
                pl.print.paper_name,
                if pl.print.landscape { "landscape" } else { "portrait" },
                format_bytes(pl.est_output_bytes)
            )
        })
    };
    ui.set_export_summary(match text { Ok(t) => t, Err(e) => format!("{e:#}") }.into());
}

/// Shows a loaded or freshly made project: settings, picture and preview.
fn show_project(ui: &AppWindow, s: &mut State, p: Project, buf: SharedPixelBuffer<Rgb8Pixel>, fill_settings: bool) {
    if fill_settings {
        show_motif(ui, &p.motif.path);
        ui.set_width_mm(format!("{}", p.grid.width_mm).into());
        ui.set_height_mm(format!("{}", p.grid.height_mm).into());
        ui.set_use_grid(false);
        ui.set_cols(p.grid.cols as i32);
        ui.set_rows(p.grid.rows as i32);
        ui.set_tile_mm(format!("{}", (p.grid.tile_mm().0 * 2.0).round() / 2.0).into());
        let si = STYLES.iter().position(|(m, _, _)| *m == p.settings.method).unwrap_or(1);
        ui.set_style_index(si as i32);
        ui.set_style_help(STYLES[si].2.into());
        ui.set_unlimited_reuse(p.settings.max_reuse.is_none());
        ui.set_max_reuse(p.settings.max_reuse.unwrap_or(3) as i32);
        ui.set_repeat_penalty(p.settings.repeat_penalty);
        ui.set_repeat_radius(p.settings.repeat_radius as i32);
        ui.set_usage_penalty(p.settings.usage_penalty);
        ui.set_crop_offsets(p.settings.crop_offsets);
        ui.set_seed(p.settings.seed as i32);
        ui.set_color_correction(p.settings.color_correction);
        ui.set_motif_blend(p.settings.motif_blend);
        show_outline(ui, p.outline.as_ref());
        let used: HashSet<i64> = p.libraries.iter().map(|l| l.id).collect();
        s.excluded = s.libs.iter().map(|l| l.id).filter(|id| !used.contains(id)).collect();
        let _ = refresh_libraries(ui, s);
    }
    let st = &p.stats;
    let (tw, th) = p.grid.tile_mm();
    ui.set_chip_photos(format!("{} different photos", thousands(st.distinct_used as u64)).into());
    ui.set_chip_distance(format!("Looks best from {:.0} m away", viewing_distance_m(tw.max(th))).into());
    ui.set_chip_repeats(format!("{} neighbours repeat", thousands(st.nearby_repeats as u64)).into());
    let mut rep = format!(
        "{}\n{} tiles, {} different photos used out of {}, each used at most {} times.\n{} times the same photo sits next to itself.\nAverage colour difference per tile: {:.1} ΔE (lower is closer).\nTimings: candidates {:.0} ms, search {:.0} ms, assignment {:.0} ms.",
        p.settings.method.title(),
        thousands(st.tiles as u64),
        thousands(st.distinct_used as u64),
        thousands(st.pool_photos as u64),
        st.max_uses,
        st.nearby_repeats,
        st.mean_delta_e,
        st.build_ms,
        st.retrieval_ms,
        st.assign_ms
    );
    if p.settings.color_correction > 0.0 || p.settings.motif_blend > 0.0 {
        rep += &format!("\nAdjusted: colour correction {:.0}%, picture blend {:.0}% (changes how your photos look).", p.settings.color_correction * 100.0, p.settings.motif_blend * 100.0);
    }
    ui.set_match_report(rep.into());
    ui.set_preview_image(Image::from_rgb8(buf));
    ui.set_has_preview(true);
    ui.set_preview_stale(false);
    ui.set_main_view(0);
    ui.set_stats_summary(format!("{} photos", thousands(st.distinct_used as u64)).into());
    ui.set_has_stats(false);
    s.stats = None;
    s.project = Some(Arc::new(p));
    update_grid_info(ui, s);
    update_export_summary(ui, s);
}

fn bars(items: &[(String, u64)]) -> Vec<Bar> {
    let max = items.iter().map(|(_, v)| *v).max().unwrap_or(0).max(1) as f32;
    items.iter().map(|(l, v)| Bar { label: l.clone().into(), value: thousands(*v).into(), frac: *v as f32 / max }).collect()
}

fn counted_chart(title: &str, note: String, items: &[Counted]) -> Chart {
    let v: Vec<(String, u64)> = items.iter().map(|c| (c.value.clone(), c.photos as u64)).collect();
    Chart { title: title.into(), note: note.into(), bars: ModelRc::new(VecModel::from(bars(&v))) }
}

fn range_note(r: &Option<Range>, f: impl Fn(f64) -> String) -> String {
    match r {
        Some(r) => format!("Typical {}, from {} to {}", f(r.median), f(r.min), f(r.max)),
        None => String::new(),
    }
}

/// Fills the statistics view: number tiles plus bar charts.
fn show_stats(ui: &AppWindow, p: &Project, st: &OutputStats) {
    let usage = p.usage();
    let mut hist = [0u64; 5];
    for &u in &usage {
        hist[(u.max(1) as usize - 1).min(4)] += 1;
    }
    let use_labels = ["Once", "2 times", "3 times", "4 times", "5 times or more"];
    let use_items: Vec<(String, u64)> = use_labels.iter().zip(hist).map(|(l, n)| (l.to_string(), n)).collect();
    let ex = &st.exif;
    let year = |d: &Option<String>| d.as_ref().and_then(|d| d.get(..4)).map(str::to_string);
    let taken = match (year(&ex.date_first), year(&ex.date_last)) {
        (Some(a), Some(b)) if a != b => format!("{a}–{b}"),
        (Some(a), _) => a,
        _ => "unknown".into(),
    };
    let figures = vec![
        Figure { label: "tiles".into(), value: thousands(st.tiles as u64).into() },
        Figure { label: "different photos".into(), value: thousands(st.distinct_photos_used as u64).into() },
        Figure { label: "photos used only once".into(), value: thousands(st.photos_used_once as u64).into() },
        Figure { label: "most uses of one photo".into(), value: format!("{}×", st.max_uses_of_one_photo).into() },
        Figure { label: "photos taken".into(), value: taken.into() },
    ];
    let charts = vec![
        Chart {
            title: "How often each photo is used".into(),
            note: format!("Number of photos. Average colour difference per tile: {:.1} ΔE", st.mean_color_error_delta_e).into(),
            bars: ModelRc::new(VecModel::from(bars(&use_items))),
        },
        counted_chart("Cameras", format!("{} of {} photos have camera info", thousands(ex.photos_with_exif as u64), thousands(st.distinct_photos_used as u64)), &ex.cameras),
        counted_chart("Aperture", range_note(&ex.aperture_f, |v| format!("f/{v:.1}")), &ex.apertures),
        counted_chart("Shutter speed", range_note(&ex.shutter_s, shutter_label), &ex.shutter_speeds),
        counted_chart("ISO", range_note(&ex.iso, |v| format!("ISO {v:.0}")), &ex.isos),
        counted_chart("Lenses", range_note(&ex.focal_length_mm, |v| format!("{v:.0} mm")), &ex.lenses),
    ];
    let (left, right): (Vec<_>, Vec<_>) = charts.into_iter().enumerate().partition(|(i, _)| i % 2 == 0);
    ui.set_stat_figures(ModelRc::new(VecModel::from(figures)));
    ui.set_stat_charts_left(ModelRc::new(VecModel::from(left.into_iter().map(|(_, c)| c).collect::<Vec<_>>())));
    ui.set_stat_charts_right(ModelRc::new(VecModel::from(right.into_iter().map(|(_, c)| c).collect::<Vec<_>>())));
    ui.set_stats_note(if ex.photos_with_exif == 0 { "None of these photos have camera info (EXIF), so only usage is shown.".into() } else { "Bars count photos, not tiles.".into() });
    ui.set_has_stats(true);
    ui.set_main_view(3);
}

fn autosave(s: &State) -> Result<()> {
    if let Some(p) = &s.project {
        p.save(&s.data_dir.join(LAST_PROJECT))?;
    }
    Ok(())
}

/// Scans the given libraries. Returns a short text about the result.
fn index_libraries(cat: &Catalog, ids: &[i64], job: &Job) -> Result<String> {
    index_libraries_counted(cat, ids, job).map(|(t, _)| t)
}

/// Like `index_libraries`, also returning how many non-photo files were ignored.
fn index_libraries_counted(cat: &Catalog, ids: &[i64], job: &Job) -> Result<(String, u64)> {
    let (mut added, mut changed, mut removed, mut skipped, mut missing, mut ignored) = (0u64, 0u64, 0u64, 0usize, 0usize, 0u64);
    for &id in ids {
        let r = update_library(cat, id, &IndexOptions::default(), job)?;
        added += r.added as u64;
        changed += r.changed as u64;
        removed += r.removed as u64;
        skipped += r.skipped.len();
        missing += r.roots_missing.len();
        ignored += r.ignored_other_files;
        if r.cancelled {
            bail!(mosaic_core::job::Cancelled);
        }
    }
    let mut t = if added + changed + removed == 0 { "Photos are up to date".to_string() } else { format!("{added} new, {changed} changed, {removed} removed photos") };
    if skipped > 0 {
        t += &format!(", {skipped} files skipped");
    }
    if missing > 0 {
        t += &format!(", {missing} folder(s) not found");
    }
    Ok((t, ignored))
}

/// Thumbnails shown when checking a picked folder.
const CHECK_THUMBS: usize = 24;
const CHECK_THUMB_PX: u32 = 200;

/// Shows the photos in a picked folder so the user can confirm it before it
/// is added. Native folder pickers hide (Windows) or grey out (macOS) the
/// files, so without this it is hard to tell if the right folder was chosen.
fn check_folder(ui: &AppWindow, st: &Shared, path: PathBuf) {
    let n = {
        let mut s = st.lock().unwrap();
        s.check_counter += 1;
        s.check = Some((s.check_counter, path.clone()));
        s.check_counter
    };
    let thumbs = Rc::new(VecModel::<Image>::default());
    ui.set_check_thumbs(ModelRc::from(thumbs.clone()));
    ui.set_check_folder(path.display().to_string().into());
    ui.set_check_summary("Looking for photos…".into());
    ui.set_check_count(0);
    ui.set_check_done(false);
    ui.set_open_step(1);
    ui.set_main_view(4);

    let weak = ui.as_weak();
    let st = st.clone();
    // True while this check is still the one on screen.
    let current = move || st.lock().unwrap().check.as_ref().map(|c| c.0) == Some(n);
    std::thread::spawn(move || {
        let mut files = Vec::new();
        let mut last = Instant::now();
        for entry in walk_images(&path).flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry.path().extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
            if ext.is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.as_str())) {
                files.push(entry.into_path());
            }
            if last.elapsed().as_millis() > 200 {
                if !current() {
                    return;
                }
                last = Instant::now();
                let found = files.len();
                let _ = weak.upgrade_in_event_loop(move |ui| ui.set_check_summary(format!("Looking for photos… {} found so far", thousands(found as u64)).into()));
            }
        }
        let total = files.len();
        let summary = match total {
            0 => "No photos found in this folder or its subfolders. Pick another folder.".to_string(),
            1 => "1 photo found (subfolders included).".to_string(),
            t if t <= CHECK_THUMBS => format!("{} photos found (subfolders included).", thousands(t as u64)),
            t => format!("{} photos found (subfolders included). Showing {CHECK_THUMBS} of them.", thousands(t as u64)),
        };
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_check_count(total as i32);
            ui.set_check_summary(summary.into());
            ui.set_check_done(true);
        });
        // Spread the samples over the whole folder, not just the first files.
        let picks: Vec<&PathBuf> = if total <= CHECK_THUMBS { files.iter().collect() } else { (0..CHECK_THUMBS).map(|i| &files[i * total / CHECK_THUMBS]).collect() };
        for p in picks {
            if !current() {
                return;
            }
            let Ok(d) = decode_file(p, Some(CHECK_THUMB_PX)) else { continue };
            let (w, h) = d.image.dimensions();
            let side = w.min(h).max(1) as f32;
            let crop = [(w as f32 - side) / 2.0 / w as f32, (h as f32 - side) / 2.0 / h as f32, side / w as f32, side / h as f32];
            let Ok(t) = resize_crop(&d.image, crop, CHECK_THUMB_PX, CHECK_THUMB_PX) else { continue };
            let buf = to_image(&t);
            let _ = weak.upgrade_in_event_loop(move |ui| {
                let m = ui.get_check_thumbs();
                if let Some(v) = m.as_any().downcast_ref::<VecModel<Image>>() {
                    v.push(Image::from_rgb8(buf));
                }
            });
        }
    });
}

/// Starts a scan of one library (after adding or finding a folder).
fn scan(ui: &AppWindow, st: &Shared, id: i64) {
    ui.set_photos_indexed(false);
    run_job(ui, st, "Scanning photos", move |job, dd| {
        let cat = Catalog::open(&dd)?;
        let (msg, ignored) = index_libraries_counted(&cat, &[id], job)?;
        Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
            s.pool = None;
            let _ = refresh_libraries(ui, s);
            update_grid_info(ui, s);
            mark_stale(ui);
            mark_indexed(ui, s, true);
            // A scan of a small folder finishes in a blink. When it finds
            // nothing, say so plainly instead of leaving the list unchanged.
            if let Some(l) = s.libs.iter().find(|l| l.id == id) {
                if l.distinct_photos == 0 {
                    let other = if ignored > 0 { format!(" {ignored} other files were ignored.") } else { String::new() };
                    ui.set_notice(
                        format!("No photos were found in this folder or its subfolders. MosaiCxM reads JPEG, PNG, WebP and TIFF files.{other}").into(),
                    );
                }
            }
            ui.set_status(msg.into());
        }))
    });
}

fn main() -> Result<()> {
    let ui = AppWindow::new()?;
    let st: Shared = Arc::new(Mutex::new(State { data_dir: data_dir(), ..Default::default() }));
    {
        let mut s = st.lock().unwrap();
        if let Err(e) = refresh_libraries(&ui, &mut s) {
            report_error(&ui, e);
        }
        apply_style(&ui, ui.get_style_index());
        update_grid_info(&ui, &s);
        if s.libs.iter().all(|l| l.distinct_photos == 0) {
            ui.set_open_step(1);
        } else {
            ui.set_open_step(2);
        }
    }

    // Startup: re-check all photo folders, then reopen the last mosaic.
    {
        let libs: Vec<i64> = st.lock().unwrap().libs.iter().filter(|l| !l.roots.is_empty()).map(|l| l.id).collect();
        run_job(&ui, &st, "Checking photo folders", move |job, dd| {
            let cat = Catalog::open(&dd)?;
            let msg = if libs.is_empty() { "Ready".to_string() } else { index_libraries(&cat, &libs, job)? };
            let last = dd.join(LAST_PROJECT);
            let reopened = if last.exists() {
                Project::load(&last).and_then(|p| {
                    let tile_px = (2400 / p.grid.cols.max(p.grid.rows)).clamp(4, 48);
                    let buf = to_image(&render_preview(&cat, &p, tile_px, job)?);
                    Ok((p, buf))
                })
                .ok()
            } else {
                None
            };
            Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                s.pool = None;
                let _ = refresh_libraries(ui, s);
                if let Some((p, buf)) = reopened {
                    // The picture can be chosen while this job runs; keep that choice.
                    let chosen = PathBuf::from(ui.get_motif_path().to_string());
                    show_project(ui, s, p, buf, true);
                    ui.set_project_name("Last mosaic (autosaved)".into());
                    if !chosen.as_os_str().is_empty() && chosen.as_path() != Path::new(ui.get_motif_path().as_str()) {
                        show_motif(ui, &chosen);
                        mark_stale(ui);
                    } else {
                        ui.set_open_step(3);
                    }
                }
                update_grid_info(ui, s);
                mark_indexed(ui, s, true);
                ui.set_status(msg.into());
            }))
        });
    }

    // ---------------- 1 Photos
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_add_folder(move || {
            let ui = w.unwrap();
            let Some(path) = rfd::FileDialog::new().set_title("Choose a folder with your photos").pick_folder() else { return };
            check_folder(&ui, &s, path);
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_confirm_folder(move || {
            let ui = w.unwrap();
            let Some(path) = s.lock().unwrap().check.take().map(|(_, p)| p) else { return };
            ui.set_main_view(0);
            let r = {
                let mut st = s.lock().unwrap();
                let r = Catalog::open(&st.data_dir).and_then(|c| {
                    let id = default_library(&c, &st.libs)?;
                    c.add_root(id, &path)?;
                    Ok(id)
                });
                let _ = refresh_libraries(&ui, &mut st);
                r
            };
            match r {
                Ok(id) => scan(&ui, &s, id),
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_cancel_folder_check(move || {
            let ui = w.unwrap();
            s.lock().unwrap().check = None;
            ui.set_main_view(0);
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_remove_folder(move |i| {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let Some((lib, path)) = s.folders.get(i as usize).cloned() else { return };
            match Catalog::open(&s.data_dir).and_then(|c| c.remove_root(lib, &path)) {
                Ok(()) => {
                    s.pool = None;
                    let _ = refresh_libraries(&ui, &mut s);
                    mark_indexed(&ui, &s, ui.get_photos_indexed());
                    update_grid_info(&ui, &s);
                    mark_stale(&ui);
                    ui.set_status(format!("Removed {} from the list. No photos were deleted.", path.display()).into());
                }
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_find_folder(move |i| {
            let ui = w.unwrap();
            let Some((lib, old)) = s.lock().unwrap().folders.get(i as usize).cloned() else { return };
            let Some(new) = rfd::FileDialog::new().set_title(format!("Where is “{}” now?", file_name(&old))).pick_folder() else { return };
            let r = {
                let mut st = s.lock().unwrap();
                let r = Catalog::open(&st.data_dir).and_then(|c| c.relink_root(lib, &old, &new, false));
                let _ = refresh_libraries(&ui, &mut st);
                r
            };
            match r {
                Ok(_) => scan(&ui, &s, lib),
                Err(e) => report_error(&ui, anyhow!("that does not look like the same folder ({e:#}). Pick the folder that has the same photos.")),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_toggle_collection(move |i, on| {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            if let Some(id) = s.libs.get(i as usize).map(|l| l.id) {
                if on {
                    s.excluded.remove(&id);
                } else {
                    s.excluded.insert(id);
                }
            }
            let _ = refresh_libraries(&ui, &mut s);
            update_grid_info(&ui, &s);
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_show_skipped(move || {
            let ui = w.unwrap();
            let s = s.lock().unwrap();
            let r = Catalog::open(&s.data_dir).and_then(|c| {
                let mut all = Vec::new();
                for id in selected_ids(&s) {
                    all.extend(c.skipped(id)?);
                }
                Ok(all)
            });
            match r {
                Ok(v) => {
                    let text = if v.is_empty() { "No skipped files.".to_string() } else { v.iter().map(|(p, w)| format!("{}\n    {w}", p.display())).collect::<Vec<_>>().join("\n") };
                    ui.set_info_title(format!("Skipped files ({})", v.len()).into());
                    ui.set_info_text(format!("These files could not be read, so they are not used. Your files are not changed.\n\n{text}").into());
                    ui.set_main_view(2);
                }
                Err(e) => report_error(&ui, e),
            }
        });
    }

    // ---------------- 2 Picture
    {
        let w = ui.as_weak();
        ui.on_browse_motif(move || {
            let ui = w.unwrap();
            if let Some(p) = rfd::FileDialog::new().set_title("Choose the picture for your mosaic").add_filter("Images", &["jpg", "jpeg", "png", "webp", "tif", "tiff"]).pick_file() {
                show_motif(&ui, &p);
                mark_stale(&ui);
                ui.set_open_step(3);
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_browse_shape(move || {
            let ui = w.unwrap();
            if let Some(p) = rfd::FileDialog::new().set_title("Choose a shape (SVG or picture)").add_filter("Shapes", &["svg", "png", "jpg", "jpeg", "webp"]).pick_file() {
                ui.set_shape_file(p.display().to_string().into());
                ui.set_shape_file_name(file_name(&p).into());
                ui.set_shape_index(6);
                mark_stale(&ui);
            } else if ui.get_shape_file().is_empty() {
                ui.set_shape_index(0);
            }
            update_grid_info(&ui, &s.lock().unwrap());
        });
    }
    {
        let w = ui.as_weak();
        ui.on_browse_suggest_folder(move || {
            if let Some(p) = rfd::FileDialog::new().set_title("Folder of pictures to try").pick_folder() {
                w.unwrap().set_suggest_folder(p.display().to_string().into());
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_suggest(move || {
            let ui = w.unwrap();
            let (libs, (grid, settings)) = match (selected_libs(&s.lock().unwrap()), read_settings(&ui)) {
                (Ok(l), Ok(x)) => (l, x),
                (Err(e), _) | (_, Err(e)) => return report_error(&ui, e),
            };
            let folder = ui.get_suggest_folder().to_string();
            let n_lib = ui.get_suggest_from_library().max(0) as usize;
            let s2 = s.clone();
            run_job(&ui, &s, "Finding good pictures", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let pool = load_pool(&s2, &cat, &libs, settings.merge_near_duplicates)?;
                let mut cands = Vec::new();
                if !folder.trim().is_empty() {
                    cands.extend(MotifCandidate::from_folder(Path::new(folder.trim()))?);
                }
                cands.extend(MotifCandidate::from_pool(&pool, n_lib, settings.seed));
                if cands.is_empty() {
                    bail!("choose a folder of pictures to try, or a number of your own photos to try");
                }
                let prev_dir = dd.join("suggestion-previews");
                let opts = RecommendOptions { trials: 8, preview_dir: Some(prev_dir), ..Default::default() };
                let recs = recommend(&cat, &pool, cands, &grid, &settings, &opts, job)?;
                let mut rows = Vec::new();
                let mut paths = Vec::new();
                for r in recs.into_iter().take(25) {
                    let path = match &r.source {
                        MotifSource::File(p) => Some(p.clone()),
                        MotifSource::Library { hash } => mosaic_core::catalog::unhex(hash).ok().and_then(|h| cat.locations(&h, &libs).ok()).and_then(|l| l.into_iter().map(|l| l.path).find(|p| p.exists())),
                    };
                    let prev = r.preview.as_ref().and_then(|p| image::open(p).ok()).map(|i| to_image(&i.to_rgb8()));
                    let pic = r.picture.as_ref().map(to_image);
                    rows.push((r.label.clone(), path.clone(), format!("{:.0}/100", r.score * 100.0), r.notes.join("; "), pic, prev));
                    paths.push(path);
                }
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    s.suggestion_paths = paths;
                    let model: Vec<SuggestionRow> = rows
                        .into_iter()
                        .map(|(label, path, score, notes, pic, prev)| SuggestionRow {
                            label: label.into(),
                            path: path.map(|p| p.display().to_string()).unwrap_or_default().into(),
                            score: score.into(),
                            notes: notes.into(),
                            picture: pic.map(Image::from_rgb8).unwrap_or_default(),
                            has_preview: prev.is_some(),
                            preview: prev.map(Image::from_rgb8).unwrap_or_default(),
                        })
                        .collect();
                    ui.set_suggest_report(format!("{} pictures ranked for {} × {} tiles. Best fit first.", model.len(), grid.cols, grid.rows).into());
                    ui.set_suggestions(ModelRc::new(VecModel::from(model)));
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_use_suggestion(move |i| {
            let ui = w.unwrap();
            let p = s.lock().unwrap().suggestion_paths.get(i as usize).cloned().flatten();
            if let Some(p) = p {
                show_motif(&ui, &p);
                mark_stale(&ui);
                ui.set_main_view(0);
                ui.set_open_step(3);
                ui.set_status("Picture chosen. Press Make preview.".into());
            }
        });
    }

    // ---------------- 3 Size & style, preview
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_style_changed(move |i| {
            let ui = w.unwrap();
            apply_style(&ui, i);
            update_grid_info(&ui, &s.lock().unwrap());
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_settings_changed(move || {
            let ui = w.unwrap();
            update_grid_info(&ui, &s.lock().unwrap());
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_generate(move || {
            let ui = w.unwrap();
            let (libs, (grid, settings)) = match (selected_libs(&s.lock().unwrap()), read_settings(&ui)) {
                (Ok(l), Ok(x)) => (l, x),
                (Err(e), _) | (_, Err(e)) => return report_error(&ui, e),
            };
            let outline = match read_outline(&ui) {
                Ok(o) => o,
                Err(e) => return report_error(&ui, e),
            };
            let motif = PathBuf::from(ui.get_motif_path().to_string());
            if motif.as_os_str().is_empty() {
                ui.set_open_step(2);
                return report_error(&ui, anyhow!("choose a picture in step 2 first"));
            }
            let s2 = s.clone();
            run_job(&ui, &s, "Making preview", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let pool = load_pool(&s2, &cat, &libs, settings.merge_near_duplicates)?;
                let (target, mref) = load_target(&motif, &grid)?;
                let p = Project::create_with_outline(&pool, libraries_refs(&cat, &libs)?, &target, mref, grid, settings, outline, job).map_err(anyhow::Error::from)?;
                let tile_px = (2400 / grid.cols.max(grid.rows)).clamp(4, 48);
                let buf = to_image(&render_preview(&cat, &p, tile_px, job)?);
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    show_project(ui, s, p, buf, false);
                    match autosave(s) {
                        Ok(()) => ui.set_project_name("New mosaic (autosaved)".into()),
                        Err(e) => report_error(ui, e.context("could not autosave")),
                    }
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_save_project(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            if let Some(path) = rfd::FileDialog::new().set_title("Save mosaic").add_filter("Mosaic project", &["json"]).set_file_name("my-mosaic.json").save_file() {
                match p.save(&path) {
                    Ok(()) => {
                        ui.set_project_name(file_name(&path).into());
                        ui.set_status(format!("Saved {}", path.display()).into());
                    }
                    Err(e) => report_error(&ui, e),
                }
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_open_project(move || {
            let ui = w.unwrap();
            let Some(path) = rfd::FileDialog::new().set_title("Open mosaic").add_filter("Mosaic project", &["json"]).pick_file() else { return };
            run_job(&ui, &s, "Opening mosaic", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let p = Project::load(&path)?;
                let tile_px = (2400 / p.grid.cols.max(p.grid.rows)).clamp(4, 48);
                let buf = to_image(&render_preview(&cat, &p, tile_px, job)?);
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    show_project(ui, s, p, buf, true);
                    let _ = autosave(s);
                    ui.set_project_name(file_name(&path).into());
                }))
            });
        });
    }

    // ---------------- 4 Export
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_export_settings_changed(move || {
            let ui = w.unwrap();
            update_export_summary(&ui, &s.lock().unwrap());
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_export(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            let manifest = ui.get_save_manifest();
            if ui.get_export_kind() == 0 {
                let fmt = raster_format(&ui);
                let plan = match num(ui.get_export_dpi(), "DPI").and_then(|dpi| plan_raster(&p, dpi, fmt)) {
                    Ok(pl) => pl,
                    Err(e) => return report_error(&ui, e),
                };
                let ext = match fmt {
                    RasterFormat::Png => "png",
                    RasterFormat::Jpeg => "jpg",
                    _ => "tif",
                };
                let Some(out) = rfd::FileDialog::new().set_title("Export mosaic image").set_file_name(format!("mosaic.{ext}")).save_file() else { return };
                run_job(&ui, &s, "Exporting image", move |job, dd| {
                    let cat = Catalog::open(&dd)?;
                    let src = resolve(&cat, &p, job)?;
                    let rep = export_raster(&p, &src, &plan, &out, job)?;
                    let details = serde_json::json!({ "format": format!("{:?}", plan.format), "dpi": plan.spec.dpi, "width_px": plan.spec.width_px, "height_px": plan.spec.height_px, "bytes": rep.bytes });
                    let (_, stats) = write_output_stats(&cat, &p, &out, "raster", details)?;
                    let mut msg = format!("Saved {} ({}) in {:.0} s.", rep.path, format_bytes(rep.bytes), rep.elapsed_ms as f64 / 1000.0);
                    if manifest {
                        msg += &save_manifest(&cat, &p, &out)?;
                    }
                    Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                        ui.set_export_report(format!("{msg} Statistics saved next to it as .stats.json.").into());
                        if s.project.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, &p)) {
                            s.stats = Some(Arc::new(stats));
                        }
                    }))
                });
            } else {
                let plan = match print_settings(&ui).and_then(|ps| plan_pdf(&p, &ps)) {
                    Ok(pl) => pl,
                    Err(e) => return report_error(&ui, e),
                };
                let Some(out) = rfd::FileDialog::new().set_title("Export print pages").add_filter("PDF", &["pdf"]).set_file_name("mosaic-print.pdf").save_file() else { return };
                run_job(&ui, &s, "Exporting PDF", move |job, dd| {
                    let cat = Catalog::open(&dd)?;
                    let src = resolve(&cat, &p, job)?;
                    let overview = render_preview(&cat, &p, (1600 / p.grid.cols.max(p.grid.rows)).clamp(2, 32), job)?;
                    let rep = export_pdf(&p, &src, Some(&overview), &plan, &out, job)?;
                    let details = serde_json::json!({ "paper": plan.print.paper_name, "landscape": plan.print.landscape, "pages": plan.printed_pages(), "overlap_mm": plan.print.overlap_mm, "dpi": plan.print.dpi, "bytes": rep.bytes });
                    let (_, stats) = write_output_stats(&cat, &p, &out, "tiled-pdf", details)?;
                    let mut msg = format!("Saved {} ({}, {} pages) in {:.0} s.", rep.path, format_bytes(rep.bytes), plan.printed_pages(), rep.elapsed_ms as f64 / 1000.0);
                    if manifest {
                        msg += &save_manifest(&cat, &p, &out)?;
                    }
                    Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                        ui.set_export_report(format!("{msg} Statistics saved next to it as .stats.json.").into());
                        if s.project.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, &p)) {
                            s.stats = Some(Arc::new(stats));
                        }
                    }))
                });
            }
        });
    }
    // ---------------- 5 Statistics
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_show_stats(move || {
            let ui = w.unwrap();
            let (p, cached) = {
                let s = s.lock().unwrap();
                (s.project.clone(), s.stats.clone())
            };
            let Some(p) = p else { return };
            if let Some(st) = cached {
                return show_stats(&ui, &p, &st);
            }
            if ui.get_busy() {
                return;
            }
            run_job(&ui, &s, "Counting statistics", move |_job, dd| {
                let cat = Catalog::open(&dd)?;
                let st = output_stats(&cat, &p, &dd.join(LAST_PROJECT), "preview", serde_json::json!({}))?;
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    if s.project.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, &p)) {
                        let st = Arc::new(st);
                        s.stats = Some(st.clone());
                        show_stats(ui, &p, &st);
                    }
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_save_stats_json(move || {
            let ui = w.unwrap();
            let (p, cached, dd) = {
                let s = s.lock().unwrap();
                (s.project.clone(), s.stats.clone(), s.data_dir.clone())
            };
            let Some(p) = p else { return report_error(&ui, anyhow!("make a preview first")) };
            let Some(out) = rfd::FileDialog::new().set_title("Save statistics").add_filter("JSON", &["json"]).set_file_name("mosaic-stats.json").save_file() else { return };
            let r = (|| -> Result<()> {
                let st = match cached {
                    Some(st) => st,
                    None => Arc::new(output_stats(&Catalog::open(&dd)?, &p, &dd.join(LAST_PROJECT), "preview", serde_json::json!({}))?),
                };
                std::fs::write(&out, serde_json::to_vec_pretty(&*st)?).with_context(|| format!("cannot write {}", out.display()))?;
                Ok(())
            })();
            match r {
                Ok(()) => ui.set_status(format!("Statistics saved to {}", out.display()).into()),
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let s = st.clone();
        ui.on_cancel(move || {
            if let Some(j) = &s.lock().unwrap().job {
                j.cancel();
            }
        });
    }
    ui.run()?;
    Ok(())
}

/// Writes `<name>-photos-used.csv` next to an export.
fn save_manifest(cat: &Catalog, p: &Project, out: &Path) -> Result<String> {
    let stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mosaic".into());
    let csv = out.with_file_name(format!("{stem}-photos-used.csv"));
    let n = write_manifest(cat, p, &csv)?;
    Ok(format!(" List of {n} photos saved to {}.", file_name(&csv)))
}
