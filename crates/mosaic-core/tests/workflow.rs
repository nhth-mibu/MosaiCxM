//! End-to-end engine tests on small generated photo sets.

use image::RgbImage;
use mosaic_core::catalog::Catalog;
use mosaic_core::export::{export_raster, plan_raster, RasterFormat};
use mosaic_core::indexer::{update_library, IndexOptions};
use mosaic_core::job::{is_cancelled, Job};
use mosaic_core::layout::{paper_by_name, PrintSettings};
use mosaic_core::matching::{GridSpec, MatchError, MatchSettings, Method, Target};
use mosaic_core::pdf::{export_pdf, plan_pdf};
use mosaic_core::pool::Pool;
use mosaic_core::project::{check_sources, libraries_refs, load_target, ProblemKind, Project};
use mosaic_core::recommend::{recommend, MotifCandidate, RecommendOptions};
use mosaic_core::render::{FileSource, Renderer, ThumbSource};
use mosaic_core::units::RasterSpec;
use std::path::{Path, PathBuf};

fn photo(i: u32) -> RgbImage {
    let c = [(i * 37 % 256) as u8, (i * 91 % 256) as u8, (i * 53 % 256) as u8];
    let (w, h) = if i % 3 == 0 { (90, 120) } else { (120, 90) };
    RgbImage::from_fn(w, h, |x, y| {
        let f = (x + y) as f32 / (w + h) as f32;
        image::Rgb([(c[0] as f32 * (0.5 + f)).min(255.0) as u8, c[1], (c[2] as f32 * (1.5 - f)).min(255.0) as u8])
    })
}

fn write_photos(dir: &Path, range: std::ops::Range<u32>) {
    std::fs::create_dir_all(dir).unwrap();
    for i in range {
        photo(i).save(dir.join(format!("p{i:04}.jpg"))).unwrap();
    }
}

struct Env {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    cat: Catalog,
}

fn env() -> Env {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let cat = Catalog::open(&root.join("data")).unwrap();
    Env { _tmp: tmp, root, cat }
}

fn opts() -> IndexOptions {
    IndexOptions { threads: 2, ..Default::default() }
}

fn motif(path: &Path) {
    RgbImage::from_fn(200, 150, |x, y| image::Rgb([(x * 255 / 200) as u8, (y * 255 / 150) as u8, 128])).save(path).unwrap();
}

fn make_project(e: &Env, libs: &[i64], s: MatchSettings, cols: u32, rows: u32) -> Project {
    let grid = GridSpec::new(200.0, 150.0, cols, rows).unwrap();
    let pool = Pool::load(&e.cat, libs, s.merge_near_duplicates).unwrap();
    let m = e.root.join("motif.png");
    motif(&m);
    let (t, mref) = load_target(&m, &grid).unwrap();
    Project::create(&pool, libraries_refs(&e.cat, libs).unwrap(), &t, mref, grid, s, &Job::default()).unwrap()
}

#[test]
fn incremental_indexing_relocation_and_corrupt_files() {
    let e = env();
    let dir = e.root.join("photos");
    write_photos(&dir, 0..20);
    std::fs::write(dir.join("broken.jpg"), b"\xFF\xD8\xFF\xE0garbage").unwrap();
    std::fs::write(dir.join("readme.txt"), b"x").unwrap();
    let lib = e.cat.create_library("L").unwrap();
    e.cat.add_root(lib, &dir).unwrap();
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!((r.added, r.decoded, r.skipped.len(), r.ignored_other_files), (21, 20, 1, 1));
    // Unchanged second run decodes nothing.
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!((r.unchanged, r.decoded, r.added), (21, 0, 0));
    // Change one, remove one, add one.
    photo(500).save(dir.join("p0001.jpg")).unwrap();
    std::fs::remove_file(dir.join("p0002.jpg")).unwrap();
    photo(21).save(dir.join("p0021.jpg")).unwrap();
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!((r.changed, r.removed, r.added, r.decoded), (1, 1, 1, 2));
    // Relocate the folder: relink keeps descriptors, nothing is decoded again.
    let moved = e.root.join("moved");
    std::fs::rename(&dir, &moved).unwrap();
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!(r.roots_missing.len(), 1);
    assert_eq!(r.removed, 0, "a missing folder must not drop records");
    e.cat.relink_root(lib, &dir, &moved, false).unwrap();
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!(r.decoded, 0);
    assert_eq!(r.removed, 0);
    // Copying files changes mtimes: they are re-hashed but descriptors are reused.
    let copy = e.root.join("copy");
    std::fs::create_dir_all(&copy).unwrap();
    for f in std::fs::read_dir(&moved).unwrap() {
        let f = f.unwrap().path();
        std::fs::copy(&f, copy.join(f.file_name().unwrap())).unwrap();
    }
    e.cat.relink_root(lib, &moved, &copy, false).unwrap();
    let r = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!(r.decoded, 0);
    assert!(r.reused_descriptors >= 20 || r.unchanged >= 20);
    // Removing the library never touches photos.
    e.cat.remove_library(lib).unwrap();
    assert!(copy.join("p0003.jpg").exists() && moved.join("p0003.jpg").exists());
}

#[test]
fn duplicates_across_libraries_count_once_and_reuse_limit_holds() {
    let e = env();
    write_photos(&e.root.join("a"), 0..30);
    write_photos(&e.root.join("b"), 20..50); // 20..30 duplicated in both
    let a = e.cat.create_library("A").unwrap();
    let b = e.cat.create_library("B").unwrap();
    e.cat.add_root(a, &e.root.join("a")).unwrap();
    e.cat.add_root(b, &e.root.join("b")).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    update_library(&e.cat, b, &opts(), &Job::default()).unwrap();
    let pool = Pool::load(&e.cat, &[a, b], false).unwrap();
    assert_eq!(pool.photos.len(), 50);
    let mut s = MatchSettings::for_method(Method::Classic);
    s.max_reuse = Some(2);
    let p = make_project(&e, &[a, b], s.clone(), 10, 10);
    assert!(p.usage().iter().all(|&u| u <= 2), "hard reuse limit");
    assert_eq!(p.tiles.len(), 100);
    // 50 distinct × 1 use cannot fill 60 tiles: infeasible, with explicit options.
    s.max_reuse = Some(1);
    let grid = GridSpec::new(200.0, 150.0, 10, 6).unwrap();
    let (t, _) = load_target(&e.root.join("motif.png"), &grid).unwrap();
    match mosaic_core::matching::match_tiles(&pool, &t, &grid, &s, &Job::default()) {
        Err(MatchError::Infeasible(inf)) => {
            assert_eq!((inf.tiles, inf.distinct_photos), (60, 50));
            assert!(inf.suggestions.iter().any(|x| x.contains("Raise the maximum reuse to 2")));
        }
        other => panic!("expected infeasible, got {:?}", other.map(|_| ())),
    }
    // Exactly at capacity (50 tiles, 50 photos, reuse 1) is feasible and every photo is used once.
    let grid = GridSpec::new(200.0, 150.0, 10, 5).unwrap();
    let (t, _) = load_target(&e.root.join("motif.png"), &grid).unwrap();
    let (tiles, stats) = mosaic_core::matching::match_tiles(&pool, &t, &grid, &s, &Job::default()).unwrap();
    assert_eq!(stats.distinct_used, 50);
    let mut seen = std::collections::HashSet::new();
    assert!(tiles.iter().all(|t| seen.insert(t.photo)));
}

#[test]
fn matching_is_deterministic() {
    let e = env();
    write_photos(&e.root.join("a"), 0..60);
    let a = e.cat.create_library("A").unwrap();
    e.cat.add_root(a, &e.root.join("a")).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    for m in Method::ALL {
        let mut s = MatchSettings::for_method(m);
        s.max_reuse = Some(3);
        let p1 = make_project(&e, &[a], s.clone(), 12, 9);
        let p2 = make_project(&e, &[a], s.clone(), 12, 9);
        assert_eq!(p1.tiles, p2.tiles, "{m:?}");
        assert_eq!(serde_json::to_string(&p1.sources).unwrap(), serde_json::to_string(&p2.sources).unwrap());
        s.seed = 99;
        let p3 = make_project(&e, &[a], s, 12, 9);
        if m == Method::Variety {
            assert_ne!(p1.tiles, p3.tiles, "seed should matter for {m:?}");
        }
    }
}

#[test]
fn rendering_is_identical_across_region_boundaries() {
    let e = env();
    write_photos(&e.root.join("a"), 0..40);
    let a = e.cat.create_library("A").unwrap();
    e.cat.add_root(a, &e.root.join("a")).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    let mut s = MatchSettings::for_method(Method::Structure);
    s.color_correction = 0.3;
    s.motif_blend = 0.2;
    let p = make_project(&e, &[a], s, 7, 5);
    let thumbs = ThumbSource::load_for(&e.cat, &p.source_hashes().unwrap()).unwrap();
    let spec = RasterSpec::new(200.0, 150.0, 97.0).unwrap(); // odd DPI: uneven tile edges
    let r = Renderer::new(&p, spec, &thumbs, 1 << 20).unwrap();
    let (w, h) = (spec.width_px, spec.height_px);
    let whole = r.render_region(0, 0, w, h, &Job::default()).unwrap();
    // Pieces with arbitrary cuts (not aligned to tiles) must match the whole exactly.
    for &(x0, y0, pw, ph) in &[(0, 0, 333, 211), (333, 0, w - 333, 211), (101, 211, 457, h - 211), (w - 7, h - 5, 7, 5)] {
        let part = r.render_region(x0, y0, pw, ph, &Job::default()).unwrap();
        for y in 0..ph {
            let a = &part[(y * pw * 3) as usize..((y + 1) * pw * 3) as usize];
            let b0 = (((y0 + y) * w + x0) * 3) as usize;
            assert_eq!(a, &whole[b0..b0 + (pw * 3) as usize], "row {y} of region at {x0},{y0}");
        }
    }
}

fn indexed_env(n: u32) -> (Env, i64) {
    let e = env();
    write_photos(&e.root.join("a"), 0..n);
    let a = e.cat.create_library("A").unwrap();
    e.cat.add_root(a, &e.root.join("a")).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    (e, a)
}

fn no_partials(dir: &Path) -> bool {
    std::fs::read_dir(dir).unwrap().all(|f| !f.unwrap().file_name().to_string_lossy().contains(mosaic_core::atomic::TEMP_MARKER))
}

#[test]
fn tiff_export_reads_back_with_independent_decoder() {
    let (e, a) = indexed_env(30);
    let p = make_project(&e, &[a], MatchSettings::for_method(Method::Classic), 9, 7);
    let chk = check_sources(&e.cat, &p, &Job::default()).unwrap();
    assert!(chk.ok());
    let src = FileSource { paths: chk.resolved.into_iter().map(Option::unwrap).collect() };
    let plan = plan_raster(&p, 120.0, RasterFormat::Tiff).unwrap();
    let out = e.root.join("m.tif");
    export_raster(&p, &src, &plan, &out, &Job::default()).unwrap();
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(std::fs::File::open(&out).unwrap())).unwrap();
    assert_eq!(dec.dimensions().unwrap(), (plan.spec.width_px as u32, plan.spec.height_px as u32));
    let img = match dec.read_image().unwrap() {
        tiff::decoder::DecodingResult::U8(v) => v,
        _ => panic!("expected u8"),
    };
    let r = Renderer::new(&p, plan.spec, &src, 64 << 20).unwrap();
    let direct = r.render_region(0, 0, plan.spec.width_px, plan.spec.height_px, &Job::default()).unwrap();
    assert_eq!(img, direct, "TIFF pixels equal the renderer output");
    // PNG export too.
    let plan = plan_raster(&p, 60.0, RasterFormat::Png).unwrap();
    export_raster(&p, &src, &plan, &e.root.join("m.png"), &Job::default()).unwrap();
    let png = image::open(e.root.join("m.png")).unwrap();
    assert_eq!((png.width() as u64, png.height() as u64), (plan.spec.width_px, plan.spec.height_px));
    let plan = plan_raster(&p, 60.0, RasterFormat::Jpeg).unwrap();
    export_raster(&p, &src, &plan, &e.root.join("m.jpg"), &Job::default()).unwrap();
    assert_eq!(image::open(e.root.join("m.jpg")).unwrap().width() as u64, plan.spec.width_px);
    assert!(no_partials(&e.root));
}

#[test]
fn cancelled_and_failed_exports_leave_nothing() {
    let (e, a) = indexed_env(30);
    let p = make_project(&e, &[a], MatchSettings::for_method(Method::Classic), 9, 7);
    let chk = check_sources(&e.cat, &p, &Job::default()).unwrap();
    let paths: Vec<PathBuf> = chk.resolved.into_iter().map(Option::unwrap).collect();
    let src = FileSource { paths: paths.clone() };
    let plan = plan_raster(&p, 200.0, RasterFormat::Tiff).unwrap();
    // Cancel after the first progress report.
    let job = Job::default();
    let j2 = job.clone();
    let job = Job::new(move |_| j2.cancel()).with_cancel_flag(job.cancel_flag());
    let out = e.root.join("c.tif");
    let err = export_raster(&p, &src, &plan, &out, &job).unwrap_err();
    assert!(is_cancelled(&err));
    assert!(!out.exists() && no_partials(&e.root));
    // A source that disappears mid-export fails the export cleanly.
    let mut bad = paths.clone();
    bad[p.tiles.last().unwrap().photo as usize] = e.root.join("gone.jpg");
    let err = export_raster(&p, &FileSource { paths: bad.clone() }, &plan, &out, &Job::default()).unwrap_err();
    assert!(err.to_string().contains("gone.jpg"), "{err}");
    assert!(!out.exists() && no_partials(&e.root));
    // Same for PDF.
    let mut ps = PrintSettings::new(paper_by_name("A4").unwrap());
    ps.dpi = 100.0;
    let pplan = plan_pdf(&p, &ps).unwrap();
    let pout = e.root.join("c.pdf");
    assert!(export_pdf(&p, &FileSource { paths: bad }, None, &pplan, &pout, &Job::default()).is_err());
    assert!(!pout.exists() && no_partials(&e.root));
    // And a successful PDF.
    export_pdf(&p, &src, None, &pplan, &pout, &Job::default()).unwrap();
    let bytes = std::fs::read(&pout).unwrap();
    assert!(bytes.starts_with(b"%PDF-1.4") && bytes.ends_with(b"%%EOF\n"));
    let pages = bytes.windows(12).filter(|w| w == b"/Type /Page ").count();
    assert_eq!(pages, pplan.layout.pages.len() + 1);
}

#[test]
fn missing_and_changed_sources_are_reported_not_substituted() {
    let (e, a) = indexed_env(25);
    let mut s = MatchSettings::for_method(Method::Classic);
    s.max_reuse = Some(1);
    let p = make_project(&e, &[a], s, 5, 4);
    let chk = check_sources(&e.cat, &p, &Job::default()).unwrap();
    assert!(chk.ok());
    let paths: Vec<PathBuf> = chk.resolved.iter().map(|p| p.clone().unwrap()).collect();
    std::fs::remove_file(&paths[0]).unwrap();
    photo(999).save(&paths[1]).unwrap();
    let chk = check_sources(&e.cat, &p, &Job::default()).unwrap();
    assert_eq!(chk.problems.len(), 2);
    assert!(matches!(chk.problems[0].kind, ProblemKind::Missing));
    assert!(matches!(chk.problems[1].kind, ProblemKind::Changed));
    assert!(chk.resolved[0].is_none() && chk.resolved[1].is_none());
    assert!(chk.report(&p).contains("never done automatically"));
    // Saved assignments are reused unchanged after a reload.
    let f = e.root.join("p.json");
    p.save(&f).unwrap();
    let q = Project::load(&f).unwrap();
    assert_eq!(q.tiles, p.tiles);
}

#[test]
fn motif_from_library_cannot_match_itself() {
    let e = env();
    let dir = e.root.join("a");
    write_photos(&dir, 0..40);
    // An exact duplicate of p0005 under another name.
    std::fs::copy(dir.join("p0005.jpg"), dir.join("copy_of_5.jpg")).unwrap();
    let a = e.cat.create_library("A").unwrap();
    e.cat.add_root(a, &dir).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    let pool = Pool::load(&e.cat, &[a], true).unwrap();
    let target_hash = *blake3::hash(&std::fs::read(dir.join("p0005.jpg")).unwrap()).as_bytes();
    let ex = pool.excluding(&target_hash, None, true);
    assert_eq!(ex.photos.len(), pool.photos.len() - 1);
    assert!(ex.index_of(&target_hash).is_none());
    // External file identical to a library photo is also excluded.
    let grid = GridSpec::new(200.0, 150.0, 8, 6).unwrap();
    let s = MatchSettings::for_method(Method::Classic);
    let cands = vec![MotifCandidate::file(&dir.join("copy_of_5.jpg")), MotifCandidate::file(&dir.join("p0007.jpg"))];
    let recs = recommend(&e.cat, &pool, cands, &grid, &s, &RecommendOptions { trials: 2, ..Default::default() }, &Job::default()).unwrap();
    assert!(recs.iter().all(|r| r.excluded_self && r.trial.is_some()));
    assert!(recs.iter().all(|r| r.score < 1.0 && !r.notes.is_empty()));
}

#[test]
fn orientation_and_crop_are_consistent_between_descriptor_and_render() {
    // A portrait photo stored landscape with EXIF orientation 6. The descriptor
    // (from indexing) and the rendered tile must agree on which side is red.
    let e = env();
    let dir = e.root.join("o");
    std::fs::create_dir_all(&dir).unwrap();
    let stored = RgbImage::from_fn(160, 80, |x, _| if x < 80 { image::Rgb([230, 20, 20]) } else { image::Rgb([20, 20, 230]) });
    let mut jpg = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut jpg, 95);
    let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    exif.extend_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0]);
    enc.add_app_segment(1, exif).unwrap();
    enc.encode(stored.as_raw(), 160, 80, jpeg_encoder::ColorType::Rgb).unwrap();
    std::fs::write(dir.join("rot.jpg"), &jpg).unwrap();
    let a = e.cat.create_library("O").unwrap();
    e.cat.add_root(a, &dir).unwrap();
    update_library(&e.cat, a, &opts(), &Job::default()).unwrap();
    let pool = Pool::load(&e.cat, &[a], false).unwrap();
    let ph = &pool.photos[0];
    assert_eq!((ph.width, ph.height), (80, 160), "upright size");
    // Upright: red on top, blue at bottom. Square crop at the top (t=0) is red.
    let top = mosaic_core::descriptor::crop_descriptor(&ph.grid, mosaic_core::descriptor::crop_rect(0.5, 1.0, 0.0));
    assert!(top[1] > 40.0, "a* of top crop should be red, got {}", top[1]);
    // Render a 1×2 grid project whose tiles use top and bottom crops.
    let grid = GridSpec::new(100.0, 200.0, 1, 2).unwrap();
    let cells = vec![[50.0, 70.0, 50.0]; 4 * 4 * 2];
    let target = Target::from_cells(1, 2, &cells);
    let mut s = MatchSettings::for_method(Method::Structure);
    s.crop_offsets = true;
    let mref = mosaic_core::project::MotifRef { path: "x".into(), content_hash: String::new(), width: 1, height: 2, crop: [0.0, 0.0, 1.0, 1.0] };
    let mut p = Project::create(&pool, vec![], &target, mref, grid, s, &Job::default()).unwrap();
    p.tiles[0].crop = mosaic_core::descriptor::crop_rect(0.5, 1.0, 0.0);
    p.tiles[1].crop = mosaic_core::descriptor::crop_rect(0.5, 1.0, 1.0);
    let src = FileSource { paths: vec![dir.join("rot.jpg")] };
    let spec = RasterSpec::with_pixels(100.0, 200.0, 40, 80).unwrap();
    let r = Renderer::new(&p, spec, &src, 1 << 20).unwrap();
    let px = r.render_region(0, 0, 40, 80, &Job::default()).unwrap();
    let at = |x: usize, y: usize| &px[(y * 40 + x) * 3..(y * 40 + x) * 3 + 3];
    assert!(at(20, 20)[0] > 180 && at(20, 20)[2] < 80, "top tile red: {:?}", at(20, 20));
    assert!(at(20, 60)[2] > 180 && at(20, 60)[0] < 80, "bottom tile blue: {:?}", at(20, 60));
}

#[test]
fn scan_is_recursive_and_skips_system_folders() {
    let e = env();
    let photos = e.root.join("photos");
    write_photos(&photos, 0..2);
    write_photos(&photos.join("2024/summer/beach/day1"), 2..5); // 4 levels deep
    write_photos(&photos.join("2025"), 5..7);
    // Not photos: NAS thumbnails, hidden folders, trash, macOS resource files.
    write_photos(&photos.join("2025/@eaDir"), 100..103);
    write_photos(&photos.join(".thumbnails"), 103..105);
    write_photos(&photos.join("$RECYCLE.BIN"), 105..106);
    photo(106).save(photos.join("2025/._p0005.jpg")).unwrap();
    #[cfg(unix)]
    {
        // A loop back to the root, and a link to a folder outside the root.
        std::os::unix::fs::symlink(&photos, photos.join("2024/loop")).unwrap();
        let outside = e.root.join("elsewhere");
        write_photos(&outside, 7..9);
        std::os::unix::fs::symlink(&outside, photos.join("linked")).unwrap();
    }
    let lib = e.cat.create_library("nested").unwrap();
    e.cat.add_root(lib, &photos).unwrap();
    let rep = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    let expected: u64 = if cfg!(unix) { 9 } else { 7 };
    assert_eq!(rep.added, expected, "{rep:?}");
    assert!(rep.walk_errors.is_empty(), "{:?}", rep.walk_errors);
    if cfg!(unix) {
        assert!(rep.skipped.iter().any(|(_, why)| why.contains("symlink loop")), "{:?}", rep.skipped);
    }
    // The same rules apply to motif folders.
    assert_eq!(MotifCandidate::from_folder(&photos).unwrap().len() as u64, expected);
    // A re-scan finds nothing new and removes nothing.
    let again = update_library(&e.cat, lib, &opts(), &Job::default()).unwrap();
    assert_eq!((again.added, again.removed), (0, 0));
}

#[test]
fn outline_limits_tiles_render_and_print_pages() {
    use mosaic_core::matching::TileAssign;
    use mosaic_core::shape::{Outline, Shape};
    let (e, a) = indexed_env(40);
    // 2 m wide strip whose right half is outside the shape (SVG draws the left half).
    let grid = GridSpec::new(2000.0, 500.0, 40, 10).unwrap();
    let svg = e.root.join("half.svg");
    std::fs::write(&svg, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100"><rect width="200" height="100"/></svg>"#).unwrap();
    let mut o = Outline::new(Shape::from_file(&svg, false).unwrap());
    o.background = [10, 20, 30];
    let pool = Pool::load(&e.cat, &[a], true).unwrap();
    let m = e.root.join("motif.png");
    motif(&m);
    let (t, mref) = load_target(&m, &grid).unwrap();
    let mut s = MatchSettings::for_method(Method::Variety);
    s.max_reuse = Some(6); // 200 placed tiles ≤ 40 × 6, but 400 would not fit
    let p = Project::create_with_outline(&pool, libraries_refs(&e.cat, &[a]).unwrap(), &t, mref, grid, s, Some(o), &Job::default()).unwrap();
    assert_eq!(p.placed_tiles(), 200);
    assert_eq!(p.stats.tiles, 200);
    for (i, tile) in p.tiles.iter().enumerate() {
        assert_eq!(tile.is_empty(), i % 40 >= 20, "tile {i}");
    }
    assert!(p.usage().iter().all(|&u| (1..=6).contains(&u)));

    // Save / load keeps the outline and the empty tiles.
    let path = e.root.join("shape.json");
    p.save(&path).unwrap();
    let q = Project::load(&path).unwrap();
    assert_eq!(q.outline, p.outline);
    assert_eq!(q.tiles, p.tiles);
    assert!(q.tiles[39] == TileAssign::EMPTY);

    // Outside the shape is painted with the background colour.
    let thumbs = ThumbSource::load_for(&e.cat, &p.source_hashes().unwrap()).unwrap();
    let spec = RasterSpec::with_pixels(2000.0, 500.0, 400, 100).unwrap();
    let r = Renderer::new(&p, spec, &thumbs, 1 << 20).unwrap();
    let px = r.render_region(0, 0, 400, 100, &Job::default()).unwrap();
    let at = |x: usize, y: usize| [px[(y * 400 + x) * 3], px[(y * 400 + x) * 3 + 1], px[(y * 400 + x) * 3 + 2]];
    assert_eq!(at(300, 50), [10, 20, 30]);
    assert_eq!(at(399, 99), [10, 20, 30]);
    assert_ne!(at(50, 50), [10, 20, 30]);

    // Print pages that lie fully outside the shape are skipped.
    let mut ps = PrintSettings::new(paper_by_name("A3").unwrap());
    ps.dpi = 30.0;
    let plan = plan_pdf(&p, &ps).unwrap();
    assert!(plan.blank_pages() > 0 && plan.printed_pages() < plan.layout.pages.len(), "{}", plan.summary());
    let out = e.root.join("shape.pdf");
    export_pdf(&p, &thumbs_source(&e, &p), None, &plan, &out, &Job::default()).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    let count = String::from_utf8_lossy(&bytes).matches("/Type /Page ").count();
    assert_eq!(count, plan.printed_pages() + 1);
}

fn thumbs_source(e: &Env, p: &Project) -> ThumbSource {
    ThumbSource::load_for(&e.cat, &p.source_hashes().unwrap()).unwrap()
}
