#!/usr/bin/env bash
# End-to-end example: generate sample photos, import them, suggest motifs,
# match, preview, save a project, export a BigTIFF + a tiled PDF + manifest.
# Everything stays inside ./e2e-demo. Usage: examples/e2e.sh [photo-count]
set -euo pipefail
N=${1:-2000}
cargo build --release -p mosaic
M="$PWD/target/release/mosaic"
D="$PWD/e2e-demo"
mkdir -p "$D"
export MOSAIC_DATA_DIR="$D/catalog"

# 1. Sample source photos (synthetic; use your own folders instead).
[ -d "$D/photos" ] || "$M" bench gen-photos "$D/photos" --count "$N"
# Target motif: a simple landscape drawn with ImageMagick-free Rust tooling is
# not available, so reuse one generated photo scaled up as a fallback motif
# unless you pass MOTIF=/path/to/image.jpg.
MOTIF=${MOTIF:-$(ls "$D"/photos/000/*.jpg | head -1)}

# 2. Library: create, add folder, index (incremental on re-runs).
"$M" library create demo 2>/dev/null || true
"$M" library add-folder demo "$D/photos" 2>/dev/null || true
"$M" library update demo
"$M" library list

# 3. Motif suggestions from the library itself (self-matches excluded).
"$M" recommend --libs demo --width-mm 800 --height-mm 600 --tile-mm 20 --method structure \
  --max-reuse 4 --from-libraries 40 --trials 4 --previews "$D/suggestions" --top 5

# 4. Match, preview and save the project.
"$M" create --motif "$MOTIF" --libs demo --width-mm 800 --height-mm 600 --tile-mm 20 \
  --method variety --max-reuse 4 --seed 7 -o "$D/project.json" --preview "$D/preview.png"

# 5. Check sources, export at 150 DPI (BigTIFF) and a tiled A4 PDF.
"$M" check "$D/project.json"
"$M" export "$D/project.json" -o "$D/mosaic.tif" --dpi 150
"$M" pdf "$D/project.json" -o "$D/print-A4.pdf" --paper A4 --landscape --margin 6 --overlap 10 --dpi 200
"$M" manifest "$D/project.json" -o "$D/photos-used.csv"
echo "Done. Outputs in $D (each export also has a .stats.json next to it)."
