//! Desktop interface (Slint). All work is done by `mosaic-core` on worker
//! threads; the UI thread only reads settings and shows results.
//!
//! The window has four colour-coded steps (Photos, Picture, Size & style,
//! Export) and a preview that is always visible. Photo libraries are handled
//! automatically: folders go into one "My photos" collection, are scanned when
//! added and re-checked on every start. The last mosaic is autosaved and
//! reopened on start.

#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{anyhow, bail, Context, Result};
use mosaic_core::catalog::{Catalog, LibraryInfo};
use mosaic_core::export::{export_raster, plan_raster, RasterFormat};
use mosaic_core::indexer::{update_library, IndexOptions};
use mosaic_core::job::{is_cancelled, Job};
use mosaic_core::layout::{PrintSettings, PAPERS};
use mosaic_core::matching::{GridSpec, MatchError, MatchSettings, Method};
use mosaic_core::pdf::{export_pdf, plan_pdf};
use mosaic_core::pool::Pool;
use mosaic_core::project::{check_sources, libraries_refs, load_target, write_manifest, write_output_stats, Project};
use mosaic_core::recommend::{recommend, MotifCandidate, MotifSource, RecommendOptions};
use mosaic_core::render::{render_preview, FileSource};
use mosaic_core::units::format_bytes;
use slint::{ComponentHandle, Image, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, VecModel};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
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
            if photos > 0 {
                if let Some(m) = s.max_reuse {
                    let need = (g.tiles() as u64).div_ceil(m as u64);
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
                "{} pages ({} across × {} down) on {} {}, plus 1 assembly page\nFile size about {}",
                pl.layout.pages.len(),
                pl.layout.cols,
                pl.layout.rows,
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
    s.project = Some(Arc::new(p));
    update_grid_info(ui, s);
    update_export_summary(ui, s);
}

fn autosave(s: &State) -> Result<()> {
    if let Some(p) = &s.project {
        p.save(&s.data_dir.join(LAST_PROJECT))?;
    }
    Ok(())
}

/// Scans the given libraries. Returns a short text about the result.
fn index_libraries(cat: &Catalog, ids: &[i64], job: &Job) -> Result<String> {
    let (mut added, mut changed, mut removed, mut skipped, mut missing) = (0u64, 0u64, 0u64, 0usize, 0usize);
    for &id in ids {
        let r = update_library(cat, id, &IndexOptions::default(), job)?;
        added += r.added as u64;
        changed += r.changed as u64;
        removed += r.removed as u64;
        skipped += r.skipped.len();
        missing += r.roots_missing.len();
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
    Ok(t)
}

/// Starts a scan of one library (after adding or finding a folder).
fn scan(ui: &AppWindow, st: &Shared, id: i64) {
    run_job(ui, st, "Scanning photos", move |job, dd| {
        let cat = Catalog::open(&dd)?;
        let msg = index_libraries(&cat, &[id], job)?;
        Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
            s.pool = None;
            let _ = refresh_libraries(ui, s);
            update_grid_info(ui, s);
            mark_stale(ui);
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
        ui.on_remove_folder(move |i| {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let Some((lib, path)) = s.folders.get(i as usize).cloned() else { return };
            match Catalog::open(&s.data_dir).and_then(|c| c.remove_root(lib, &path)) {
                Ok(()) => {
                    s.pool = None;
                    let _ = refresh_libraries(&ui, &mut s);
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
            if let Some(p) = rfd::FileDialog::new().set_title("Choose the picture for your mosaic").add_filter("Images", &["jpg", "jpeg", "png", "webp"]).pick_file() {
                show_motif(&ui, &p);
                mark_stale(&ui);
                ui.set_open_step(3);
            }
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
                    rows.push((r.label.clone(), path.clone(), format!("{:.0}/100", r.score * 100.0), r.notes.join("; "), prev));
                    paths.push(path);
                }
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    s.suggestion_paths = paths;
                    let model: Vec<SuggestionRow> = rows
                        .into_iter()
                        .map(|(label, path, score, notes, prev)| SuggestionRow {
                            label: label.into(),
                            path: path.map(|p| p.display().to_string()).unwrap_or_default().into(),
                            score: score.into(),
                            notes: notes.into(),
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
                let p = Project::create(&pool, libraries_refs(&cat, &libs)?, &target, mref, grid, settings, job).map_err(anyhow::Error::from)?;
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
                    write_output_stats(&cat, &p, &out, "raster", details)?;
                    let mut msg = format!("Saved {} ({}) in {:.0} s.", rep.path, format_bytes(rep.bytes), rep.elapsed_ms as f64 / 1000.0);
                    if manifest {
                        msg += &save_manifest(&cat, &p, &out)?;
                    }
                    Ok(Box::new(move |ui: &AppWindow, _s: &mut State| ui.set_export_report(msg.into())))
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
                    let details = serde_json::json!({ "paper": plan.print.paper_name, "landscape": plan.print.landscape, "pages": plan.layout.pages.len(), "overlap_mm": plan.print.overlap_mm, "dpi": plan.print.dpi, "bytes": rep.bytes });
                    write_output_stats(&cat, &p, &out, "tiled-pdf", details)?;
                    let mut msg = format!("Saved {} ({}, {} pages) in {:.0} s.", rep.path, format_bytes(rep.bytes), plan.layout.pages.len(), rep.elapsed_ms as f64 / 1000.0);
                    if manifest {
                        msg += &save_manifest(&cat, &p, &out)?;
                    }
                    Ok(Box::new(move |ui: &AppWindow, _s: &mut State| ui.set_export_report(msg.into())))
                });
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
