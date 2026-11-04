#!/bin/sh
# Adds MosaiCxM to your app menu (current user only, no sudo).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/256x256/apps"
cp "$here/mosaic-gui" "$HOME/.local/bin/"
cp "$here/mosaicxm.png" "$HOME/.local/share/icons/hicolor/256x256/apps/mosaicxm.png"
sed "s|^Exec=.*|Exec=$HOME/.local/bin/mosaic-gui|" "$here/mosaicxm.desktop" > "$HOME/.local/share/applications/mosaicxm.desktop"
echo "MosaiCxM installed. Look for it in your app menu."
