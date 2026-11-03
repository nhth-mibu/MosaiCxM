# Benchmarks

All numbers below were measured on 2026-11-03 with release builds. Nothing here is
an estimate unless it says so. Peak memory is the peak resident set size (RSS)
reported by `scripts/peakrss.py`.

## Hardware

- Cloud VM, 4 logical CPUs, Intel Xeon @ 2.80 GHz, x86_64, 15 GB RAM, Linux.
- 4 worker threads (the default: one per logical CPU).
- Local SSD-backed disk. Files were in the OS page cache for most runs, so
  cold-disk import on a laptop with a spinning or network drive will be slower.

## Dataset

**This is synthetic data, not a real camera library.**

- 100,000 JPEGs, 640 × 480 px, 5.0 GB, made with
  `mosaic bench gen-photos DIR --count 100000 --seed 1` (random two-colour
  gradients with shapes). Generating them took 359.6 s.
- Motif: `work/motif.jpg`, an ordinary photo.
- Output size: 3000 × 3000 mm, 20 mm mosaic tiles, so a 150 × 150 grid
  (22,500 tiles).

## Reproduce

```sh
cargo build --release
export MOSAIC_DATA_DIR=$PWD/bench-catalog
M=target/release/mosaic
P="python3 scripts/peakrss.py"

$M bench gen-photos bench100k --count 100000 --seed 1
$M library create big && $M library add-folder big bench100k
$P $M library update big                      # first import
$P $M library update big                      # unchanged re-scan

$P $M create --motif motif.jpg --libs big --width-mm 3000 --height-mm 3000 \
   --tile-mm 20 --method structure --max-reuse 2 -o big.json --preview big.png
$P $M recommend --libs big --width-mm 3000 --height-mm 3000 --tile-mm 20 \
   --method structure --max-reuse 2 --from-libraries 200 --trials 6 --top 3
$P $M export big.json --dpi 300 --format tiff -o big.tif
$P $M pdf big.json --paper A1 --dpi 300 -o big.pdf

$P $M bench match --photos 100000 --cols 150 --rows 150 --seed 1 --brute-tiles 200
```

## Results

### Indexing (100,000 real JPEG files)

| Run | Wall time | Peak RSS | Notes |
|---|---|---|---|
| First import | 117.3 s | 53 MB | 856 photos/s decoded; folder scan 0.47 s |
| Re-scan, nothing changed | 1.5 s | 45 MB | size + mtime check only |
| Re-scan, 1,000 files touched (mtime changed, content same) | 1.9 s | – | re-hashed, 1,000 cached descriptors reused, 0 decoded |

### Matching (100,000-photo pool, 22,500 tiles)

| Method | Settings | Wall time | Peak RSS | Distinct photos | Adjacent repeats | Mean ΔE |
|---|---|---|---|---|---|---|
| Classic | unlimited reuse | 1.5 s | 163 MB | 384 | 61,629 | 1.3 |
| Structure-aware | max reuse 2 | 13.2 s | 226 MB | 11,322 | 114 | 13.8 |
| Variety-aware | max reuse 2 | 23.7 s | 325 MB | 11,773 | 4 | 12.2 |

Wall time includes loading the pool from SQLite (1.2–2.0 s). Most of the
structure and variety time is assignment under the reuse limit (9.0 s and
20.2 s). The structure run also wrote a 3600 × 3600 px preview in 1.05 s.

The classic run shows why reuse limits matter: unlimited reuse with a
synthetic library picks a few hundred photos many times over.

### Candidate retrieval: k-d tree vs. exhaustive search

From `bench match` (synthetic descriptors, 200 tiles, single thread):

| Method | k-d tree | Exhaustive | Speed-up | Same best candidate |
|---|---|---|---|---|
| Classic (3-D mean Lab) | 0.01 ms/tile | 3.27 ms/tile | 356× | 100% |
| Structure (12-D layout) | 0.03 ms/tile | 4.31 ms/tile | 148× | 100% |

An earlier run on the same machine measured 176–758×. Treat the speed-up as
"two orders of magnitude", not as an exact figure.

### Motif suggestions

200 library photos screened, 6 low-resolution trial mosaics (77 × 77 tiles),
top 3 reported: **17.6 s, 293 MB peak RSS.**

### Large export: 3 m × 3 m at 300 DPI

| Output | Size | Wall time | Peak RSS | Verified with |
|---|---|---|---|---|
| Tiled BigTIFF, Deflate, 35,433 × 35,433 px | 1.48 GB | 56.5 s | 217 MB | `tifffile` (BigTIFF, shape, 256 px tiles, 300 DPI) |
| Tiled print PDF, A1 portrait, 24 pages + overview | 259 MB | 54.0 s | 210 MB | `pdfinfo` (A1 pages), `pdfimages` (300 ppi page images) |

The raw RGB raster would be 3.77 GB. Peak memory stayed near 220 MB because the
renderer works on bands and the writers stream to disk.

### End-to-end example

`examples/e2e.sh 500` (500 generated photos, 800 × 600 mm, A4 PDF) runs every
step from import to manifest in 3.9 s.

## Not verified

- Real camera libraries: large (12 MP and up) JPEGs, mixed PNG/WebP, real EXIF,
  ICC profiles from real cameras, slow or network drives. Import of 12 MP JPEGs
  will be slower per photo than the 640 × 480 files here, although DCT scaling
  means the decoder does not decode at full size.
- Matching quality on a real 100,000-photo collection. The synthetic photos
  are gradients, so the ΔE values above say little about how a real mosaic
  will look.
- Printing on a real printer. Page sizes and image resolution were checked in
  the PDF; physical alignment was not.
- Other platforms. All runs were on Linux. The desktop app was tested under
  Xvfb on Linux only.
