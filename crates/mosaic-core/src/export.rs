//! Bounded-memory raster export (tiled BigTIFF, streaming PNG, streaming JPEG).
//!
//! The full raster is never held in memory: the renderer produces horizontal
//! bands that are encoded and written immediately. All exports go through a
//! temporary file that is renamed only after success.

use crate::atomic::{ensure_disk_space, TempOutput};
use crate::job::Job;
use crate::project::Project;
use crate::render::{Renderer, TileSource};
use crate::tiff::{TiffCompression, TiledTiffWriter, TILE};
use crate::units::{format_bytes, RasterSpec};
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::cell::RefCell;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RasterFormat {
    Tiff,
    TiffUncompressed,
    Png,
    Jpeg,
}

impl RasterFormat {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "tif" | "tiff" | "bigtiff" => Self::Tiff,
            "tiff-raw" | "tiff-uncompressed" => Self::TiffUncompressed,
            "png" => Self::Png,
            "jpg" | "jpeg" => Self::Jpeg,
            _ => bail!("unknown format '{s}' (tiff, tiff-raw, png, jpeg)"),
        })
    }
    pub fn from_path(p: &Path) -> Option<Self> {
        let e = p.extension()?.to_str()?.to_ascii_lowercase();
        Self::parse(&e).ok()
    }
}

pub const JPEG_MAX_SIDE: u64 = 65_535;
/// PNG is streamed, but many viewers fail far below the format limit.
pub const PNG_MAX_SIDE: u64 = 1 << 31;

#[derive(Debug, Clone, Serialize)]
pub struct RasterPlan {
    pub spec: RasterSpec,
    pub format: RasterFormat,
    pub tiles: usize,
    pub tile_mm: (f64, f64),
    pub tile_px: (u64, u64),
    pub band_rows: u64,
    pub cache_budget: u64,
    pub est_peak_memory: u64,
    pub est_output_bytes: u64,
    pub est_temp_disk: u64,
    pub notes: Vec<String>,
}

impl RasterPlan {
    pub fn summary(&self) -> String {
        let s = &self.spec;
        let mut out = format!(
            "Output: {} × {} px ({:.0} × {:.0} mm at {} DPI), {:.2} gigapixels, {} raw RGB\n\
             Tiles: {} ({} × {}), each about {:.1} × {:.1} mm / {} × {} px\n\
             Estimated peak memory: {} (band {} rows + tile cache {})\n\
             Estimated output size: {} (written to a temporary file of the same size, then renamed)\n",
            s.width_px,
            s.height_px,
            s.width_mm,
            s.height_mm,
            s.dpi,
            s.pixel_count().unwrap_or(0) as f64 / 1e9,
            format_bytes(s.rgb_bytes().unwrap_or(0)),
            self.tiles,
            0,
            0,
            self.tile_mm.0,
            self.tile_mm.1,
            self.tile_px.0,
            self.tile_px.1,
            format_bytes(self.est_peak_memory),
            self.band_rows,
            format_bytes(self.cache_budget),
            format_bytes(self.est_output_bytes),
        );
        out = out.replace(" (0 × 0)", "");
        for n in &self.notes {
            out += &format!("Note: {n}\n");
        }
        out
    }
}

pub fn cache_budget_for(spec: &RasterSpec, project: &Project) -> u64 {
    let row_bytes = spec.width_px * spec.height_px.div_ceil(project.grid.rows as u64) * 3;
    (row_bytes * 2).clamp(128 << 20, 1 << 30)
}

pub fn plan_raster(project: &Project, dpi: f64, format: RasterFormat) -> Result<RasterPlan> {
    let g = &project.grid;
    let spec = RasterSpec::new(g.width_mm, g.height_mm, dpi)?;
    let mut notes = Vec::new();
    match format {
        RasterFormat::Jpeg if spec.width_px > JPEG_MAX_SIDE || spec.height_px > JPEG_MAX_SIDE => {
            bail!("JPEG is limited to {JPEG_MAX_SIDE} px per side; this export is {}×{} px. Use TIFF or a lower DPI.", spec.width_px, spec.height_px)
        }
        RasterFormat::Png if spec.width_px > PNG_MAX_SIDE || spec.height_px > PNG_MAX_SIDE => bail!("PNG side limit exceeded"),
        RasterFormat::Png if spec.pixel_count()? > 500_000_000 => {
            notes.push("PNG is written in one stream (single-threaded compression) and many viewers cannot open images this large; BigTIFF is recommended.".into())
        }
        RasterFormat::Tiff | RasterFormat::TiffUncompressed => notes.push(
            "Tiled BigTIFF (256 px tiles). Readable by libtiff-based tools, GIMP, Photoshop, Affinity, QGIS/GDAL; some simple viewers do not support BigTIFF.".into(),
        ),
        _ => {}
    }
    notes.push("Colour: sRGB, 8 bits per channel. No CMYK conversion or printer ICC profile is applied.".into());
    let band_rows = match format {
        RasterFormat::Tiff | RasterFormat::TiffUncompressed => TILE as u64,
        _ => 256,
    };
    let cache = cache_budget_for(&spec, project);
    let band = spec.width_px * band_rows * 3;
    let threads = rayon::current_num_threads() as u64;
    let tile_px = (spec.width_px.div_ceil(g.cols as u64), spec.height_px.div_ceil(g.rows as u64));
    let decode_ws = threads * (tile_px.0 * tile_px.1 * 3 * 8).max(4 << 20);
    let est_peak_memory = cache + band * 3 + decode_ws + (64 << 20);
    let raw = spec.rgb_bytes()?;
    let est_output_bytes = match format {
        RasterFormat::TiffUncompressed => crate::tiff::uncompressed_size(spec.width_px as u32, spec.height_px as u32),
        RasterFormat::Tiff => raw * 6 / 10,
        RasterFormat::Png => raw * 7 / 10,
        RasterFormat::Jpeg => raw / 8,
    };
    Ok(RasterPlan {
        spec,
        format,
        tiles: g.tiles(),
        tile_mm: g.tile_mm(),
        tile_px,
        band_rows,
        cache_budget: cache,
        est_peak_memory,
        est_output_bytes,
        est_temp_disk: est_output_bytes,
        notes,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub path: String,
    pub bytes: u64,
    pub elapsed_ms: u128,
    pub width_px: u64,
    pub height_px: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

pub fn export_raster(project: &Project, source: &dyn TileSource, plan: &RasterPlan, out: &Path, job: &Job) -> Result<ExportReport> {
    let t0 = Instant::now();
    ensure_disk_space(out, plan.est_temp_disk)?;
    let renderer = Renderer::new(project, plan.spec, source, plan.cache_budget)?;
    let mut tmp = TempOutput::create(out)?;
    let (w, h) = (plan.spec.width_px, plan.spec.height_px);
    match plan.format {
        RasterFormat::Tiff | RasterFormat::TiffUncompressed => {
            let comp = if plan.format == RasterFormat::Tiff { TiffCompression::Deflate } else { TiffCompression::None };
            let mut tw = TiledTiffWriter::new(BufWriter::with_capacity(1 << 20, tmp.file()), w as u32, h as u32, plan.spec.dpi, comp)?;
            let across = tw.tiles_across() as u64;
            let mut y = 0u64;
            while y < h {
                job.check()?;
                let bh = (TILE as u64).min(h - y);
                let band = renderer.render_region(0, y, w, bh, job)?;
                let encoded: Vec<Result<Vec<u8>>> = (0..across)
                    .into_par_iter()
                    .map(|tx| {
                        let tile = cut_tile(&band, w, bh, tx * TILE as u64);
                        TiledTiffWriter::<BufWriter<&mut std::fs::File>>::encode_tile(comp, &tile)
                    })
                    .collect();
                for e in encoded {
                    tw.write_encoded(&e?)?;
                }
                y += bh;
                job.report("exporting TIFF", y, h);
            }
            tw.finish()?.flush()?;
        }
        RasterFormat::Png => {
            let bw = BufWriter::with_capacity(1 << 20, tmp.file());
            let mut enc = png::Encoder::new(bw, w as u32, h as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_compression(png::Compression::Fast);
            let ppm = (plan.spec.dpi / 0.0254).round() as u32;
            enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
            let mut writer = enc.write_header()?;
            {
                let mut sw = writer.stream_writer()?;
                let mut y = 0u64;
                while y < h {
                    job.check()?;
                    let bh = 256u64.min(h - y);
                    let band = renderer.render_region(0, y, w, bh, job)?;
                    sw.write_all(&band)?;
                    y += bh;
                    job.report("exporting PNG", y, h);
                }
                sw.finish()?;
            }
        }
        RasterFormat::Jpeg => {
            let bw = BufWriter::with_capacity(1 << 20, tmp.file());
            write_jpeg_region(bw, &renderer, 0, 0, w, h, 92, plan.spec.dpi, job, "exporting JPEG")?;
        }
    }
    job.check()?;
    let bytes = tmp.file().metadata()?.len();
    tmp.commit()?;
    let st = renderer.stats();
    Ok(ExportReport {
        path: out.display().to_string(),
        bytes,
        elapsed_ms: t0.elapsed().as_millis(),
        width_px: w,
        height_px: h,
        cache_hits: st.cache_hits,
        cache_misses: st.cache_misses,
    })
}

/// Cut a TILE×TILE tile starting at column `x0` from a band, padding by edge replication.
fn cut_tile(band: &[u8], w: u64, bh: u64, x0: u64) -> Vec<u8> {
    let t = TILE as usize;
    let mut out = vec![0u8; t * t * 3];
    for ty in 0..t {
        let sy = (ty as u64).min(bh - 1) as usize;
        for tx in 0..t {
            let sx = (x0 + tx as u64).min(w - 1) as usize;
            let s = (sy * w as usize + sx) * 3;
            out[(ty * t + tx) * 3..(ty * t + tx) * 3 + 3].copy_from_slice(&band[s..s + 3]);
        }
    }
    out
}

/// Stream a JPEG of a global pixel region, rendering bands on demand.
#[allow(clippy::too_many_arguments)]
pub fn write_jpeg_region<W: Write>(w: W, r: &Renderer, x0: u64, y0: u64, width: u64, height: u64, quality: u8, dpi: f64, job: &Job, stage: &str) -> Result<()> {
    if width > JPEG_MAX_SIDE || height > JPEG_MAX_SIDE || width == 0 || height == 0 {
        bail!("JPEG region {width}×{height} px outside 1–{JPEG_MAX_SIDE} px");
    }
    struct Src<'a, 'b> {
        r: &'a Renderer<'b>,
        x0: u64,
        y0: u64,
        w: u64,
        h: u64,
        band: &'a RefCell<(u64, u64, Vec<u8>)>,
        err: &'a RefCell<Option<anyhow::Error>>,
        job: &'a Job,
        stage: &'a str,
    }
    impl jpeg_encoder::ImageBuffer for Src<'_, '_> {
        fn get_jpeg_color_type(&self) -> jpeg_encoder::JpegColorType {
            jpeg_encoder::JpegColorType::Ycbcr
        }
        fn width(&self) -> u16 {
            self.w as u16
        }
        fn height(&self) -> u16 {
            self.h as u16
        }
        fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
            let y = y as u64;
            let mut b = self.band.borrow_mut();
            if self.err.borrow().is_none() && !(y >= b.0 && y < b.0 + b.1) {
                let start = y - y % 256;
                let bh = 256.min(self.h - start);
                match self.r.render_region(self.x0, self.y0 + start, self.w, bh, self.job) {
                    Ok(v) => *b = (start, bh, v),
                    Err(e) => *self.err.borrow_mut() = Some(e),
                }
                self.job.report(self.stage, start + bh, self.h);
            }
            if self.err.borrow().is_some() {
                for _ in 0..self.w {
                    buffers[0].push(128);
                    buffers[1].push(128);
                    buffers[2].push(128);
                }
                return;
            }
            let row = &b.2[((y - b.0) * self.w * 3) as usize..((y - b.0 + 1) * self.w * 3) as usize];
            for p in row.chunks_exact(3) {
                let (yy, cb, cr) = jpeg_encoder::rgb_to_ycbcr(p[0], p[1], p[2]);
                buffers[0].push(yy);
                buffers[1].push(cb);
                buffers[2].push(cr);
            }
        }
    }
    let band = RefCell::new((0, 0, Vec::new()));
    let err = RefCell::new(None);
    let src = Src { r, x0, y0, w: width, h: height, band: &band, err: &err, job, stage };
    let mut enc = jpeg_encoder::Encoder::new(w, quality);
    let d = dpi.round().clamp(1.0, 65535.0) as u16;
    enc.set_density(jpeg_encoder::PixelDensity { density: (d, d), unit: jpeg_encoder::PixelDensityUnit::Inches });
    enc.encode_image(src).map_err(|e| anyhow!("JPEG encoding failed: {e}"))?;
    if let Some(e) = err.into_inner() {
        return Err(e);
    }
    Ok(())
}
