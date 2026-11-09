# MosaiCxM technical guide

Everything about building, the command-line tool, the engine and its limits. For how to use the app, see the [README](../README.md).

| Term | Meaning |
|---|---|
| **Target motif** | The image the mosaic should resemble. |
| **Source photos** | The photographs used to build it, taken from one or more libraries. |
| **Mosaic tiles** | Individual photo placements in the grid. |
| **Print pages** | Sheets of paper that are assembled into the physical mosaic. |

## Quick start

Requirements: Rust 1.85+ (2024 edition). On Linux the desktop app also needs the usual X11/Wayland libraries (`libxkbcommon-x11`, `libfontconfig`). The CLI has no system dependencies.

```bash
cargo build --release              # builds the `mosaic` CLI and `mosaic-gui`
target/release/mosaic-gui          # desktop app
examples/e2e.sh                    # full CLI workflow on generated sample photos
cargo test --release               # engine tests
```

The catalogue (index, descriptors, thumbnails) lives in the platform data folder, e.g. `~/.local/share/mosaic` on Linux. Override it with `--data-dir` or `MOSAIC_DATA_DIR`.

## Releases

Pushing a tag like `v0.1.0` runs `.github/workflows/release.yml` and publishes a GitHub Release with:

- `mosaic-<tag>-windows-x64.zip` (`mosaic-gui.exe` only)
- `mosaic-<tag>-macos-arm64.dmg` (`MosaiCxM.app` only)
- `mosaic-<tag>-linux-x64.tar.gz` (`mosaic-gui`, `install.sh`, desktop file and icon)

The app icon is drawn by `scripts/make_icons.py` (Pillow). Re-run it after changing the design; it rewrites `crates/mosaic-gui/ui/icon.png` and `crates/mosaic-gui/icons/`.

```bash
git tag v0.1.0 && git push origin v0.1.0
```

You can also run the workflow by hand from the Actions tab; it then only uploads the files as workflow artifacts. The macOS app is ad-hoc signed but not notarized, so on first launch right-click it and choose **Open**.

## CLI workflow

```bash
mosaic library create travel
mosaic library add-folder travel ~/Pictures/Travel
mosaic library add-folder travel /Volumes/Archive/2019
mosaic library update travel            # incremental; re-run any time
mosaic library list                     # counts, folders, missing folders
mosaic library skipped travel           # corrupt/unsupported files and why
mosaic library relink travel /Volumes/Archive/2019 /Volumes/NewDisk/2019   # moved drive
mosaic library remove-folder travel ~/Pictures/Travel   # catalogue only; photos untouched
mosaic library remove travel            # catalogue only; photos untouched

mosaic methods                          # plain-language method explanations

# Motif suggestions (from a folder of candidates and/or photos in the libraries)
mosaic recommend --libs travel,family --width-mm 1000 --height-mm 700 --tile-mm 15 \
  --method structure --max-reuse 3 --folder ~/motif-ideas --from-libraries 100 \
  --previews ./suggestions

# Match and save a project (+ fast preview from thumbnails)
mosaic create --motif portrait.jpg --libs travel,family --width-mm 1000 --height-mm 700 \
  --tile-mm 15 --method variety --max-reuse 3 --repeat-penalty 8 --seed 42 \
  -o portrait.json --preview preview.png

# Non-rectangular outline: built-in shape, a shape file, or the motif's own transparency.
#   SVG: drawn parts are inside. PNG/WebP with transparency: opaque parts.
#   Other images: dark parts. The shape is centred and fitted inside the artwork
#   (--shape motif covers the artwork like the motif, so it lines up with it).
mosaic create --motif portrait.jpg --libs travel --width-mm 1000 --height-mm 1000 --tile-mm 15 \
  --shape heart --background '#ffffff' --shape-threshold 0.5 -o heart.json
mosaic create ... --shape logo.svg [--shape-invert] -o logo.json

mosaic preview portrait.json -o preview.png --tile-px 32
mosaic check portrait.json              # missing/changed source files, before export
mosaic export portrait.json -o portrait.tif --dpi 300 --dry-run   # estimates only
mosaic export portrait.json -o portrait.tif --dpi 300             # tiled BigTIFF
mosaic pdf portrait.json -o print.pdf --paper A3 --landscape --margin 6 --overlap 10 --dpi 300
mosaic pdf portrait.json -o print.pdf --paper 610x914 --margin 5,5,12,5   # custom paper, per-side margins
mosaic manifest portrait.json -o photos-used.csv
```

Every export also writes `<output>.stats.json`. It records the tile count, distinct photos used, the reuse spread, libraries and method, plus an EXIF summary of the photos used: aperture, shutter, ISO and focal-length ranges with medians, the most common values, cameras, lenses and date range. Photos without EXIF are counted separately. The manifest CSV lists every photo used with its path, use count, attribution fields (EXIF Artist/Copyright or an `attribution.csv` in the source folder) and EXIF camera settings. Usage rights are never inferred.

## Desktop app

`mosaic-gui` is built with [Slint](https://slint.dev) and calls the same engine. See the [README](../README.md) for how to use it. Implementation notes:

- One window with five colour-coded steps on the left and the preview on the right. The UI is in `crates/mosaic-gui/ui/app.slint`; colours are in its `Theme` global and follow the system light/dark setting.
- **Automatic photo handling.** Folders are added to a collection named "My photos" (or to the only library, if exactly one exists from the CLI). Adding a folder starts an index update right away. On start the app runs an index update for every library with folders (an unchanged 100k library takes about 1.5 s), so there is no Update button. A missing folder shows a "Find it…" action, which calls `relink_root` and then updates. Named libraries made with the CLI appear under "Choose collections" when there is more than one.
- **Style presets** map to methods and reuse limits: Sharp = Classic, max 5 uses; Balanced = Structure-aware, max 3; Varied = Variety-aware, max 2. All other matching settings are under Advanced.
- **Autosave.** Each new preview is saved to `last-project.json` in the data folder and reopened on start. "Save as…" writes a copy anywhere.
- **Export** estimates update automatically from `plan_raster` / `plan_pdf`. The optional photo list is written as `<name>-photos-used.csv` next to the export.
- **Statistics** (step 5) shows `output_stats` for the current project as bar charts: photo-use histogram from `Project::usage`, and the top-10 cameras, lenses, apertures, shutter speeds and ISOs (counted per distinct photo). It is computed on a worker thread the first time the step opens and cached until the next preview; an export replaces it with the stats it wrote. "Save as JSON…" writes the same `OutputStats` JSON as the `<output>.stats.json` files that exports keep writing.
- Long jobs run on worker threads. The status bar shows progress and has a Cancel button. Errors appear in a banner above the preview.

## Mosaic methods

All three share descriptors, candidate retrieval and assignment code. They differ in scoring and assignment rules.

- **A · Classic colour.** Scores the CIELAB distance between the tile's mean colour and the photo crop's mean colour. It is the fastest and reproduces large colour areas well.
- **B · Structure-aware.** Compares a 4×4 Lab layout per tile plus light/dark gradients, so photos with a matching edge or gradient win. Outlines and local detail come out better.
- **C · Variety-aware.** Uses the structure score plus a per-use penalty and a nearby-repeat penalty, and also tries off-centre crops. More different photos appear, at a slightly higher colour error.

Controls shared by all methods:

- **Max reuse** is a hard limit per photo, or unlimited. It applies across all selected libraries, and exact duplicates count as one photo. Near-duplicates (same perceptual hash, aspect and colour) are merged by default.
- **Repeat penalty and radius** discourage the same photo appearing within N tiles.
- **Seed.** Tiles are assigned in a seeded random order. The PRNG is implemented in-tree, so seeds stay reproducible across dependency upgrades.
- **Colour correction** (0–50% of each tile's mean error, capped at 15 ΔE) and **motif blend** (0–50% opacity) are off by default. Both are disclosed in the CLI output, the UI and the project.

If the settings can't be met, for example 22,500 tiles with max reuse 1 but only 10,000 photos, matching stops with the reason and explicit options: raise the reuse limit, use a coarser grid (with a suggested size), or add N photos. When the nearest candidates are used up, the search widens to an index of photos that still have capacity. A narrow search therefore never causes a false "infeasible".

**Recognisability vs. photo visibility.** Smaller tiles, colour correction and motif blend make the motif easier to recognise. Bigger tiles and no adjustments keep each photo true to itself. As a rough guide, tiles of 10–15 mm read as a picture from about 2 m. 30 mm tiles need 4–5 m, but each photo is clearly visible up close. Motif suggestions warn when much of the motif's detail is smaller than a tile.

## Motif suggestions

Candidates come from a separate folder and/or photos in the selected libraries. Ranking has two passes:

1. **Cheap screening of every candidate.** It uses a thumbnail or reduced decode and measures colour coverage (nearest pool photo per area), repetition pressure (demand per Lab colour bin versus distinct photos × reuse level), contrast, and the share of luminance detail finer than one tile.
2. **Trial mosaics for the top N.** These run real matching on descriptors (downscaled to about 6,000 tiles if larger) and measure achieved ΔE, edge preservation and variety. Each trial includes a preview.

The notes explain the result in plain words, for example "strong colour coverage", "limited dark blue photos may cause weaker matches under the reuse limit" or "much fine detail will disappear". When a library photo, or an external file identical to one, is evaluated as a motif, that photo, its exact duplicates and near-identical copies are removed from the sources first. The scores estimate **technical** suitability for this collection and these settings. They do not measure artistic quality. Re-run after changing libraries or settings, because a motif that works with one collection may not work with another.

## Libraries and indexing

- Formats: JPEG, PNG, WebP and TIFF (incl. BigTIFF), detected from file content. Files are only picked up by extension, case-insensitive (`jpg jpeg jpe jfif png webp tif tiff`). 16-bit PNG/TIFF is reduced to 8 bits per channel; alpha is composited over white. Other files are counted as ignored.
- Folders are scanned at any depth. Symlinked folders are followed; a symlink loop is reported once and not scanned twice. If a folder can't be read, files already indexed under it are kept rather than marked removed.
- Skipped: hidden files and folders (names starting with `.`, which covers macOS `._` files, `.Trashes` and `.Spotlight-V100`) and system or NAS folders (`@eaDir`, `#recycle`, `#snapshot`, `$RECYCLE.BIN`, `System Volume Information`, `lost+found`). Motif folders use the same rules.
- **EXIF orientation** is applied before analysis, preview and export, so crops always refer to the upright image.
- **Change detection:** path + size + mtime (ns). On a mismatch the file is re-hashed (BLAKE3). If a current descriptor already exists for that content (a moved, copied or touched file, or a duplicate), it is reused without decoding. Only new content is decoded. *Limitation:* an edit that keeps both size and mtime is missed unless you run `library update --verify`, which re-hashes everything.
- Removed files are dropped from the index. A **missing source folder** (for example an unplugged drive) is reported, and its records are kept.
- **Relinking** changes a folder's location and checks a sample of files first. Descriptors are keyed by content hash, so nothing is re-analysed.
- Corrupt, truncated, oversized (>512 MB) or undecodable files are reported with a reason and never fail the import.
- **Bounded resources.** A fixed worker pool works through batches of 128 files. JPEGs are decoded with DCT scaling (to ~128 px for analysis), decode allocations are capped at 768 MB, and thumbnails are 96 px JPEGs.
- **Stored per photo:** a 16×16 Lab grid of the whole frame (768 B), a 64-bit dHash, the upright size, and a 96 px thumbnail. Descriptors carry `DESCRIPTOR_VERSION`. When it is bumped, they are recomputed on the next update. EXIF fields carry a separate `META_VERSION`, which triggers a re-read without decoding.
- **Crop/aspect.** Because the grid covers the uncropped frame, the descriptor for any tile aspect and crop position is derived on the fly. Photos are never decoded again for matching or previews.
- **Crop position.** Wide photos are cut from the centre. Tall photos are cut near the top (`TALL_CROP_T` = 0.2 along the free height), because heads are near the top of most portraits. With off-centre crops on, wide photos also try the left and right ends, and tall photos also try the very top, but never the bottom.
- **Storage split.** Source photos stay in their folders and are never modified. Cached analysis is in `catalog.sqlite` in the data folder. Projects are JSON files saved wherever you choose.

## Projects

A project stores the motif reference (path, content hash, crop), the selected libraries, the grid and physical size, all matching settings and the seed, the descriptor version, source identities (content hashes), the final tile assignments with crop rectangles and colour adjustments, the target cells and match statistics. Previews and exports always reuse the saved assignment, so changing DPI or paper never re-matches.

**Outline (optional).** `outline` holds the shape (built-in, or a file path with its content hash and the invert/cover flags), the background colour and the edge threshold. The shape is sampled 6×6 times per tile (2×2 above 250k tiles); a tile is placed when its inside share is at least the threshold. Tiles outside are stored as `{"p": 4294967295}` (empty), are skipped by matching, reuse limits and statistics, and render as the background colour (motif blend is not applied there). Edges follow tile borders, so no photo is cut. PDF pages whose window holds no placed tile are left out and drawn grey on the assembly map. Older projects without `outline` load unchanged.

Before an export, every source is resolved to an existing, unchanged file. A changed stat triggers a re-hash. Missing, changed or unindexed sources are listed with the number of affected tiles and the fix (relink, restore, update index). The export does not start, and photos are never silently substituted.

## Large output and printing

- **Pixel maths.** Sizes are computed with checked integer arithmetic from mm and DPI. 3,000 mm at 300 DPI is 35,433 px per side, 1.26 Gpx, 3.77 GB raw RGB. Tile edges are `floor(i × W_px / cols)`, so they are exact and identical in every strip and page.
- **Rendering** works in bounded bands. Each tile is resampled once at its full global pixel size into a byte-bounded LRU cache, then cut. Neighbouring strips and pages therefore share identical pixels, and page boundaries can't create seams (tested). The full raster never exists in memory.
- **BigTIFF** is tiled 256×256 with Deflate and a horizontal predictor, written by a small in-tree streaming writer. The `tiff` crate's encoder writes strips only. PNG is streamed row by row. JPEG is streamed through `jpeg-encoder`'s row-pull API, and is limited to 65,535 px per side by the format.
- **Tiled PDF.** One JPEG image per print page at the print DPI, streamed straight into the file. Only object offsets are kept in memory. Pages include crop marks, red alignment ticks for the overlap, the page label (row letter + column) with neighbours, artwork coordinates, a 50 mm ruler and an "actual size" reminder. Page 1 is an assembly map with instructions, a 50 × 50 mm square and a 100 mm ruler. The PDF asks viewers for `PrintScaling /None`.
- **Page layout.** Usable area = paper − margins − 2 × mark space. Settings that leave no usable area are rejected. Pages step by `usable − overlap`, so overlap is duplicated content and never extra artwork. Edge pages are partially filled. After trimming, the assembled size equals the requested size exactly (tested). Page images are snapped to the global pixel grid, so there is no scaling drift.
- **Before exporting,** the dry run or Estimate button shows pixel and physical size, tile count and tile size, estimated peak memory, estimated output/temp size, format notes and page count.
- **Safe writes.** Exports are written to a hidden temp file in the target folder and renamed only after `fsync`. Cancellation, errors and a full disk all remove the temp file. Free disk space is checked against the estimate first.

**Printing checklist:** print at 100% / "Actual size", turn off "Fit to page", "Shrink oversized pages" and borderless printing, then measure the 50 mm square before printing everything.

## Colour handling

- Matching happens in CIELAB (D65, CIE76 ΔE). Averages are taken in linear light.
- Embedded RGB ICC profiles are converted to sRGB with `moxcms`. Untagged images are treated as sRGB. Gray and CMYK profiles are not colour-managed; CMYK JPEGs get a naive conversion.
- Transparency is composited over white (paper).
- The same decode, orientation, crop and resize path (Catmull-Rom) is used for previews and exports.
- Output is 8-bit sRGB. There is **no** printer ICC or CMYK conversion, and no professional print-fidelity claim is made.

## Dependencies and trade-offs

| Need | Choice | Why / trade-off |
|---|---|---|
| Decode | `image` (PNG, WebP, TIFF), `jpeg-decoder` (JPEG with DCT scaling) | DCT scaling makes indexing and export decode only what is needed. `jpeg-decoder` is in maintenance mode but stable. |
| Resize | `fast_image_resize` | SIMD convolution; much faster than `image::imageops`. |
| EXIF | `kamadak-exif` (+ `image` orientation parser) | Pure Rust; reads JPEG/PNG/WebP/TIFF containers. |
| ICC | `moxcms` | Pure Rust CMS (also used by `image`); avoids a C `lcms2` build. |
| Database | `rusqlite` (bundled SQLite) | One local file, transactional, no server. |
| Hash | `blake3` | Fast full-content hashing for change and duplicate detection. |
| Nearest neighbours | `kiddo` k-d tree | Measured roughly 150–750× faster than exhaustive search at 100k photos with the same best candidate (see benchmarks). |
| Parallelism | `rayon`, one global pool | Decode, match and render share it, so they don't oversubscribe the CPU. |
| Encoders | `png`, `jpeg-encoder`, `flate2`; in-tree BigTIFF and PDF writers | Streaming with bounded memory. `pdf-writer` keeps the whole document in memory. |
| SVG shapes | `resvg` (no text/fonts features) | Pure Rust SVG renderer; only used to rasterise outline masks. |
| UI | Slint (Fluent style), `rfd` file dialogs | Polished, Rust-native, no web stack. Slint's royalty-free licence requires attribution in the app or docs. |

## Benchmarks

See [docs/BENCHMARKS.md](BENCHMARKS.md) for commands, hardware, dataset and results. They separate what was measured from what is estimated.

## Tests

`cargo test --release -p mosaic-core` covers:

- mm/DPI conversion and checked dimensions
- EXIF orientation and crop consistency between descriptor and render
- incremental indexing, relocation and corrupt files
- duplicates across combined libraries
- deterministic matching
- hard reuse limits and infeasible settings
- motif self-exclusion
- page coverage, overlap, margins and exact sizes
- identical pixels across region boundaries
- missing and changed sources
- cancellation, failed exports and temp-file cleanup
- BigTIFF read back with the independent `tiff` decoder

## Known limitations

- Performance was measured on synthetic photos and descriptors, not on a real 100,000-photo camera library (see benchmarks).
- Change detection relies on size + mtime unless `--verify` is used.
- No CMYK or printer-profile output. Gray/CMYK ICC profiles are not colour-managed.
- Assignment is greedy in seeded random order, not globally optimal. Under very tight reuse limits on huge grids it slows down (seconds at 22,500 tiles × 100k photos).
- BigTIFF is not supported by some simple image viewers. Use GIMP, Photoshop, Affinity, or libtiff/GDAL tools.
- Online image search and downloading are out of scope.

## Licence notes

The desktop UI uses [Slint](https://slint.dev) under its Royalty-free Desktop licence, which requires the "Made with Slint" attribution shown in the app's status bar. Choose a licence for this repository before distributing it.
