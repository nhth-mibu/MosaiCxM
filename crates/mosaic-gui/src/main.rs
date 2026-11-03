//! Desktop interface (Slint). All work is done by `mosaic-core` on worker
//! threads; the UI thread only reads settings and shows results.

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

#[derive(Default)]
struct State {
    data_dir: PathBuf,
    libs: Vec<LibraryInfo>,
    selected: HashSet<i64>,
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
    {
        let mut s = st.lock().unwrap();
        s.job = Some(job.clone());
    }
    ui.set_busy(true);
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
                    apply(&ui, &mut s);
                    ui.set_status(format!("{label} finished in {secs:.1} s").into());
                }
                Err(e) if is_cancelled(&e) => ui.set_status(format!("{label} cancelled. Nothing was written.").into()),
                Err(e) => {
                    let msg = match e.downcast_ref::<MatchError>() {
                        Some(MatchError::Infeasible(inf)) => format!("{}\nOptions:\n  • {}", inf.message, inf.suggestions.join("\n  • ")),
                        _ => format!("{e:#}"),
                    };
                    ui.set_status(format!("{label} failed: {}", msg.lines().next().unwrap_or("")).into());
                    ui.set_match_report(msg.clone().into());
                    ui.set_export_report(msg.clone().into());
                    ui.set_library_report(msg.into());
                }
            }
        });
    });
}

fn refresh_libraries(ui: &AppWindow, st: &mut State) -> Result<()> {
    let cat = Catalog::open(&st.data_dir)?;
    st.libs = cat.libraries()?;
    let ids: HashSet<i64> = st.libs.iter().map(|l| l.id).collect();
    st.selected.retain(|i| ids.contains(i));
    let rows: Vec<LibraryRow> = st
        .libs
        .iter()
        .map(|l| LibraryRow {
            name: l.name.clone().into(),
            summary: format!("{} photos · {} files · {} skipped · {} folder(s)", l.distinct_photos, l.files_ok, l.files_skipped, l.roots.len()).into(),
            folders: if l.roots.is_empty() {
                "No folders yet. Add one below.".to_string()
            } else {
                l.roots.iter().map(|r| format!("{}{}", r.path.display(), if r.path.is_dir() { "" } else { "   [missing: reconnect or relink]" })).collect::<Vec<_>>().join("\n")
            }
            .into(),
            selected: st.selected.contains(&l.id),
        })
        .collect();
    if ui.get_current_library() >= rows.len() as i32 {
        ui.set_current_library(rows.len() as i32 - 1);
    }
    ui.set_libraries(ModelRc::new(VecModel::from(rows)));
    Ok(())
}

fn current_lib(ui: &AppWindow, st: &State) -> Result<i64> {
    let i = ui.get_current_library();
    st.libs.get(i.max(0) as usize).filter(|_| i >= 0).map(|l| l.id).ok_or_else(|| anyhow!("select a library first"))
}

fn num(s: SharedString, what: &str) -> Result<f64> {
    s.trim().replace(',', ".").parse::<f64>().with_context(|| format!("{what} must be a number"))
}

fn read_settings(ui: &AppWindow) -> Result<(GridSpec, MatchSettings)> {
    let (w, h) = (num(ui.get_width_mm(), "width")?, num(ui.get_height_mm(), "height")?);
    let grid = if ui.get_use_tile_size() {
        GridSpec::from_tile_mm(w, h, num(ui.get_tile_mm(), "tile size")?)?
    } else {
        GridSpec::new(w, h, ui.get_cols().max(1) as u32, ui.get_rows().max(1) as u32)?
    };
    let method = Method::ALL[ui.get_method_index().clamp(0, 2) as usize];
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

fn apply_method_defaults(ui: &AppWindow, i: i32) {
    let m = Method::ALL[i.clamp(0, 2) as usize];
    let s = MatchSettings::for_method(m);
    ui.set_method_help(m.explanation().into());
    ui.set_repeat_penalty(s.repeat_penalty);
    ui.set_repeat_radius(s.repeat_radius as i32);
    ui.set_usage_penalty(s.usage_penalty);
    ui.set_crop_offsets(s.crop_offsets);
}

fn update_grid_info(ui: &AppWindow, st: &State) {
    let info = match read_settings(ui) {
        Ok((g, s)) => {
            let photos: u64 = st.libs.iter().filter(|l| st.selected.contains(&l.id)).map(|l| l.distinct_photos).sum();
            let (tw, th) = g.tile_mm();
            let mut t = format!("{} × {} = {} mosaic tiles, each {:.1} × {:.1} mm.", g.cols, g.rows, g.tiles(), tw, th);
            if photos > 0 {
                t += &format!(" Selected libraries: up to {photos} source photos.");
                if let Some(m) = s.max_reuse {
                    if photos * m as u64 + 0 < g.tiles() as u64 {
                        t += &format!(" Not enough photos for max {m} use(s): needs at least {} distinct photos.", (g.tiles() as u64).div_ceil(m as u64));
                    }
                }
            }
            t
        }
        Err(e) => format!("{e}"),
    };
    ui.set_grid_info(info.into());
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
    let mut v: Vec<i64> = st.libs.iter().filter(|l| st.selected.contains(&l.id)).map(|l| l.id).collect();
    v.sort_unstable();
    if v.is_empty() {
        bail!("tick at least one library on the Source photos tab");
    }
    Ok(v)
}

fn print_settings(ui: &AppWindow) -> Result<PrintSettings> {
    let i = ui.get_paper_index().max(0) as usize;
    let mut ps = if i < PAPERS.len() { PrintSettings::new(PAPERS[i]) } else { PrintSettings::custom(num(ui.get_custom_w(), "paper width")?, num(ui.get_custom_h(), "paper height")?) };
    ps.landscape = ui.get_landscape();
    ps.margins_mm = [num(ui.get_margin_mm(), "margin")?; 4];
    ps.overlap_mm = num(ui.get_overlap_mm(), "overlap")?;
    ps.dpi = num(ui.get_pdf_dpi(), "PDF DPI")?;
    ps.marks = ui.get_marks();
    Ok(ps)
}

fn raster_format(ui: &AppWindow) -> RasterFormat {
    [RasterFormat::Tiff, RasterFormat::Png, RasterFormat::Jpeg][ui.get_raster_format_index().clamp(0, 2) as usize]
}

fn report_error(ui: &AppWindow, e: anyhow::Error) {
    ui.set_status(format!("{e:#}").into());
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
        Ok(d) => ui.set_motif_image(Image::from_rgb8(to_image(&d.image))),
        Err(e) => ui.set_status(format!("Cannot open motif: {e}").into()),
    }
}

fn main() -> Result<()> {
    let ui = AppWindow::new()?;
    let st: Shared = Arc::new(Mutex::new(State { data_dir: data_dir(), ..Default::default() }));
    {
        let mut s = st.lock().unwrap();
        if let Err(e) = refresh_libraries(&ui, &mut s) {
            report_error(&ui, e);
        }
        // Tick all libraries initially.
        s.selected = s.libs.iter().map(|l| l.id).collect();
        let _ = refresh_libraries(&ui, &mut s);
        if !s.libs.is_empty() {
            ui.set_current_library(0);
        }
        update_grid_info(&ui, &s);
    }
    apply_method_defaults(&ui, ui.get_method_index());
    ui.set_status(format!("Catalogue: {}", st.lock().unwrap().data_dir.display()).into());

    // ---------------- libraries
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_create_library(move || {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let r = Catalog::open(&s.data_dir).and_then(|c| c.create_library(&ui.get_new_library_name()));
            match r {
                Ok(id) => {
                    s.selected.insert(id);
                    ui.set_new_library_name("".into());
                    let _ = refresh_libraries(&ui, &mut s);
                    if let Some(i) = s.libs.iter().position(|l| l.id == id) {
                        ui.set_current_library(i as i32);
                    }
                }
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_remove_library(move || {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let r = current_lib(&ui, &s).and_then(|id| Catalog::open(&s.data_dir)?.remove_library(id));
            match r {
                Ok(()) => {
                    s.pool = None;
                    let _ = refresh_libraries(&ui, &mut s);
                    ui.set_library_report("Library removed from the catalogue. No photos were deleted.".into());
                }
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_select_library(move |i| {
            let ui = w.unwrap();
            ui.set_current_library(i);
            let s = s.lock().unwrap();
            if let Some(l) = s.libs.get(i as usize) {
                if let Some(r) = l.roots.first() {
                    ui.set_relink_old(r.path.display().to_string().into());
                }
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_toggle_library(move |i, on| {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            if let Some(id) = s.libs.get(i as usize).map(|l| l.id) {
                if on {
                    s.selected.insert(id);
                } else {
                    s.selected.remove(&id);
                }
            }
            let _ = refresh_libraries(&ui, &mut s);
            update_grid_info(&ui, &s);
        });
    }
    {
        let w = ui.as_weak();
        ui.on_browse_folder(move || {
            if let Some(p) = rfd::FileDialog::new().set_title("Choose a folder of source photos").pick_folder() {
                w.unwrap().set_folder_path(p.display().to_string().into());
            }
        });
    }
    for remove in [false, true] {
        let (w, s) = (ui.as_weak(), st.clone());
        let f = move || {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let path = PathBuf::from(ui.get_folder_path().to_string());
            let r = current_lib(&ui, &s).and_then(|id| {
                let c = Catalog::open(&s.data_dir)?;
                if remove { c.remove_root(id, &path) } else { c.add_root(id, &path).map(|_| ()) }
            });
            match r {
                Ok(()) => {
                    s.pool = None;
                    let _ = refresh_libraries(&ui, &mut s);
                    ui.set_library_report(if remove { "Folder removed (photos untouched).".into() } else { "Folder added. Press Update index to analyse its photos.".into() });
                }
                Err(e) => report_error(&ui, e),
            }
        };
        if remove { ui.on_remove_folder(f) } else { ui.on_add_folder(f) }
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_relink(move || {
            let ui = w.unwrap();
            let mut s = s.lock().unwrap();
            let r = current_lib(&ui, &s).and_then(|id| {
                Catalog::open(&s.data_dir)?.relink_root(id, Path::new(ui.get_relink_old().as_str()), Path::new(ui.get_relink_new().as_str()), false)
            });
            match r {
                Ok(rep) => {
                    let _ = refresh_libraries(&ui, &mut s);
                    ui.set_library_report(format!("Relinked: {}/{} sampled files found. Press Update index to pick up any changes.", rep.found, rep.sampled).into());
                }
                Err(e) => report_error(&ui, e),
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_update_index(move || {
            let ui = w.unwrap();
            let id = match current_lib(&ui, &s.lock().unwrap()) {
                Ok(i) => i,
                Err(e) => return report_error(&ui, e),
            };
            run_job(&ui, &s, "Updating index", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let r = update_library(&cat, id, &IndexOptions::default(), job)?;
                let mut t = format!(
                    "{} files: {} added, {} changed, {} removed, {} unchanged.\nDecoded {} new photos, reused {} cached analyses, {} duplicate files, {} skipped, {} non-image files ignored.\nTime {:.1} s.",
                    r.files_seen, r.added, r.changed, r.removed, r.unchanged, r.decoded, r.reused_descriptors, r.duplicate_files, r.skipped.len(), r.ignored_other_files, r.elapsed_ms as f64 / 1000.0
                );
                for (p, why) in r.skipped.iter().take(30) {
                    t += &format!("\nskipped {}: {why}", p.display());
                }
                for m in &r.roots_missing {
                    t += &format!("\nfolder missing (records kept): {}", m.display());
                }
                if r.cancelled {
                    t += "\nCancelled. Photos processed so far are saved; update again to continue.";
                }
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    s.pool = None;
                    let _ = refresh_libraries(ui, s);
                    update_grid_info(ui, s);
                    ui.set_library_report(t.into());
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_show_skipped(move || {
            let ui = w.unwrap();
            let s = s.lock().unwrap();
            let r = current_lib(&ui, &s).and_then(|id| Catalog::open(&s.data_dir)?.skipped(id));
            match r {
                Ok(v) if v.is_empty() => ui.set_library_report("No skipped files.".into()),
                Ok(v) => ui.set_library_report(v.iter().map(|(p, w)| format!("{}: {w}", p.display())).collect::<Vec<_>>().join("\n").into()),
                Err(e) => report_error(&ui, e),
            }
        });
    }

    // ---------------- motif & suggestions
    {
        let w = ui.as_weak();
        ui.on_browse_motif(move || {
            let ui = w.unwrap();
            if let Some(p) = rfd::FileDialog::new().set_title("Choose the target motif").add_filter("Images", &["jpg", "jpeg", "png", "webp"]).pick_file() {
                ui.set_motif_path(p.display().to_string().into());
                show_motif(&ui, &p);
            }
        });
    }
    {
        let w = ui.as_weak();
        ui.on_motif_entered(move || {
            let ui = w.unwrap();
            let p = PathBuf::from(ui.get_motif_path().trim().to_string());
            show_motif(&ui, &p);
        });
    }
    {
        let w = ui.as_weak();
        ui.on_browse_suggest_folder(move || {
            if let Some(p) = rfd::FileDialog::new().set_title("Folder of candidate motifs").pick_folder() {
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
            run_job(&ui, &s, "Suggesting motifs", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let pool = load_pool(&s2, &cat, &libs, settings.merge_near_duplicates)?;
                let mut cands = Vec::new();
                if !folder.trim().is_empty() {
                    cands.extend(MotifCandidate::from_folder(Path::new(folder.trim()))?);
                }
                cands.extend(MotifCandidate::from_pool(&pool, n_lib, settings.seed));
                if cands.is_empty() {
                    bail!("choose a folder of candidate motifs or a number of library photos to try");
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
                    ui.set_suggest_report(format!("{} suggestions for {} × {} tiles. Scores estimate technical suitability for these libraries and settings, not artistic quality. Change libraries or settings and suggest again to re-rank.", model.len(), grid.cols, grid.rows).into());
                    ui.set_suggestions(ModelRc::new(VecModel::from(model)));
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_use_suggestion(move |i| {
            let ui = w.unwrap();
            if let Some(Some(p)) = s.lock().unwrap().suggestion_paths.get(i as usize) {
                ui.set_motif_path(p.display().to_string().into());
                show_motif(&ui, p);
                ui.set_status("Motif selected. Generate a preview on the settings tab.".into());
            }
        });
    }

    // ---------------- settings & preview
    {
        let w = ui.as_weak();
        ui.on_method_changed(move |i| {
            let ui = w.unwrap();
            apply_method_defaults(&ui, i);
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
                return report_error(&ui, anyhow!("choose a target motif first"));
            }
            let s2 = s.clone();
            run_job(&ui, &s, "Matching and preview", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let pool = load_pool(&s2, &cat, &libs, settings.merge_near_duplicates)?;
                let (target, mref) = load_target(&motif, &grid)?;
                let p = Project::create(&pool, libraries_refs(&cat, &libs)?, &target, mref, grid, settings, job).map_err(anyhow::Error::from)?;
                let tile_px = (2400 / grid.cols.max(grid.rows)).clamp(4, 48);
                let img = render_preview(&cat, &p, tile_px, job)?;
                let buf = to_image(&img);
                let st = &p.stats;
                let mut rep = format!(
                    "{} · {} tiles · {} distinct photos used (pool {}), max {} uses, {} adjacent repeats, mean colour error ΔE {:.1}.\nTimings: candidates {:.0} ms, retrieval {:.0} ms, assignment {:.0} ms.",
                    p.settings.method.title(), st.tiles, st.distinct_used, st.pool_photos, st.max_uses, st.nearby_repeats, st.mean_delta_e, st.build_ms, st.retrieval_ms, st.assign_ms
                );
                if p.settings.color_correction > 0.0 || p.settings.motif_blend > 0.0 {
                    rep += &format!("\nAdjusted: colour correction {:.0}%, motif blend {:.0}% (changes how source photos look).", p.settings.color_correction * 100.0, p.settings.motif_blend * 100.0);
                }
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    s.project = Some(Arc::new(p));
                    ui.set_preview_image(Image::from_rgb8(buf));
                    ui.set_has_preview(true);
                    ui.set_project_path("".into());
                    ui.set_match_report(rep.into());
                }))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_save_project(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            if let Some(path) = rfd::FileDialog::new().set_title("Save mosaic project").add_filter("Mosaic project", &["json"]).set_file_name("mosaic-project.json").save_file() {
                match p.save(&path) {
                    Ok(()) => ui.set_project_path(path.display().to_string().into()),
                    Err(e) => report_error(&ui, e),
                }
            }
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_open_project(move || {
            let ui = w.unwrap();
            let Some(path) = rfd::FileDialog::new().set_title("Open mosaic project").add_filter("Mosaic project", &["json"]).pick_file() else { return };
            run_job(&ui, &s, "Opening project", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let p = Project::load(&path)?;
                let tile_px = (2400 / p.grid.cols.max(p.grid.rows)).clamp(4, 48);
                let buf = to_image(&render_preview(&cat, &p, tile_px, job)?);
                Ok(Box::new(move |ui: &AppWindow, s: &mut State| {
                    ui.set_motif_path(p.motif.path.display().to_string().into());
                    show_motif(ui, &p.motif.path);
                    ui.set_width_mm(format!("{}", p.grid.width_mm).into());
                    ui.set_height_mm(format!("{}", p.grid.height_mm).into());
                    ui.set_use_tile_size(false);
                    ui.set_cols(p.grid.cols as i32);
                    ui.set_rows(p.grid.rows as i32);
                    ui.set_method_index(Method::ALL.iter().position(|m| *m == p.settings.method).unwrap_or(0) as i32);
                    ui.set_method_help(p.settings.method.explanation().into());
                    ui.set_unlimited_reuse(p.settings.max_reuse.is_none());
                    ui.set_max_reuse(p.settings.max_reuse.unwrap_or(3) as i32);
                    ui.set_repeat_penalty(p.settings.repeat_penalty);
                    ui.set_repeat_radius(p.settings.repeat_radius as i32);
                    ui.set_usage_penalty(p.settings.usage_penalty);
                    ui.set_crop_offsets(p.settings.crop_offsets);
                    ui.set_seed(p.settings.seed as i32);
                    ui.set_color_correction(p.settings.color_correction);
                    ui.set_motif_blend(p.settings.motif_blend);
                    s.selected = p.libraries.iter().map(|l| l.id).collect();
                    let _ = refresh_libraries(ui, s);
                    ui.set_preview_image(Image::from_rgb8(buf));
                    ui.set_has_preview(true);
                    ui.set_project_path(path.display().to_string().into());
                    ui.set_match_report(format!("Loaded saved assignment: {} tiles, {} photos. Exports reuse it without re-matching.", p.tiles.len(), p.sources.len()).into());
                    s.project = Some(Arc::new(p));
                }))
            });
        });
    }

    // ---------------- export
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_estimate_raster(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            let r = num(ui.get_export_dpi(), "DPI").and_then(|dpi| plan_raster(&p, dpi, raster_format(&ui)));
            ui.set_export_report(match r { Ok(plan) => plan.summary(), Err(e) => format!("{e:#}") }.into());
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_estimate_pdf(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            let r = print_settings(&ui).and_then(|ps| plan_pdf(&p, &ps));
            ui.set_export_report(match r { Ok(plan) => plan.summary(), Err(e) => format!("{e:#}") }.into());
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_export_raster(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            let fmt = raster_format(&ui);
            let plan = match num(ui.get_export_dpi(), "DPI").and_then(|dpi| plan_raster(&p, dpi, fmt)) {
                Ok(pl) => pl,
                Err(e) => return ui.set_export_report(format!("{e:#}").into()),
            };
            let ext = match fmt { RasterFormat::Png => "png", RasterFormat::Jpeg => "jpg", _ => "tif" };
            let Some(out) = rfd::FileDialog::new().set_title("Export full mosaic").set_file_name(format!("mosaic.{ext}")).save_file() else { return };
            ui.set_export_path(out.display().to_string().into());
            ui.set_export_report(plan.summary().into());
            run_job(&ui, &s, "Exporting image", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let src = resolve(&cat, &p, job)?;
                let rep = export_raster(&p, &src, &plan, &out, job)?;
                let details = serde_json::json!({ "format": format!("{:?}", plan.format), "dpi": plan.spec.dpi, "width_px": plan.spec.width_px, "height_px": plan.spec.height_px, "bytes": rep.bytes });
                let (sp, st) = write_output_stats(&cat, &p, &out, "raster", details)?;
                let msg = format!(
                    "{}\nExported {} ({}) in {:.1} s.\nStatistics: {} distinct photos, EXIF on {}; saved {}",
                    plan.summary(), rep.path, format_bytes(rep.bytes), rep.elapsed_ms as f64 / 1000.0, st.distinct_photos_used, st.exif.photos_with_exif, sp.display()
                );
                Ok(Box::new(move |ui: &AppWindow, _s: &mut State| ui.set_export_report(msg.into())))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_export_pdf(move || {
            let ui = w.unwrap();
            let Some(p) = s.lock().unwrap().project.clone() else { return };
            let plan = match print_settings(&ui).and_then(|ps| plan_pdf(&p, &ps)) {
                Ok(pl) => pl,
                Err(e) => return ui.set_export_report(format!("{e:#}").into()),
            };
            let Some(out) = rfd::FileDialog::new().set_title("Export print PDF").add_filter("PDF", &["pdf"]).set_file_name("mosaic-print.pdf").save_file() else { return };
            ui.set_pdf_path(out.display().to_string().into());
            ui.set_export_report(plan.summary().into());
            run_job(&ui, &s, "Exporting PDF", move |job, dd| {
                let cat = Catalog::open(&dd)?;
                let src = resolve(&cat, &p, job)?;
                let overview = render_preview(&cat, &p, (1600 / p.grid.cols.max(p.grid.rows)).clamp(2, 32), job)?;
                let rep = export_pdf(&p, &src, Some(&overview), &plan, &out, job)?;
                let details = serde_json::json!({ "paper": plan.print.paper_name, "landscape": plan.print.landscape, "pages": plan.layout.pages.len(), "overlap_mm": plan.print.overlap_mm, "dpi": plan.print.dpi, "bytes": rep.bytes });
                let (sp, st) = write_output_stats(&cat, &p, &out, "tiled-pdf", details)?;
                let msg = format!(
                    "{}\nExported {} ({}, {} print pages + overview) in {:.1} s.\nStatistics: {} distinct photos, EXIF on {}; saved {}",
                    plan.summary(), rep.path, format_bytes(rep.bytes), plan.layout.pages.len(), rep.elapsed_ms as f64 / 1000.0, st.distinct_photos_used, st.exif.photos_with_exif, sp.display()
                );
                Ok(Box::new(move |ui: &AppWindow, _s: &mut State| ui.set_export_report(msg.into())))
            });
        });
    }
    {
        let (w, s) = (ui.as_weak(), st.clone());
        ui.on_export_manifest(move || {
            let ui = w.unwrap();
            let (p, dd) = {
                let s = s.lock().unwrap();
                (s.project.clone(), s.data_dir.clone())
            };
            let Some(p) = p else { return };
            let Some(out) = rfd::FileDialog::new().set_title("Export manifest").add_filter("CSV", &["csv"]).set_file_name("photos-used.csv").save_file() else { return };
            match Catalog::open(&dd).and_then(|c| write_manifest(&c, &p, &out)) {
                Ok(n) => ui.set_export_report(format!("Manifest of {n} photos written to {}. Usage rights are not inferred; check each photo's licence.", out.display()).into()),
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
