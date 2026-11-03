//! Physical units and checked pixel arithmetic.
//!
//! One global artwork coordinate system is used everywhere: millimetres from
//! the top-left corner of the artwork. Pixel positions are always derived from
//! the *global* pixel size of the artwork so that strips, pages and tiles that
//! are rendered separately share identical pixel edges.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const MM_PER_INCH: f64 = 25.4;
/// Largest supported artwork side (3 m).
pub const MAX_ARTWORK_MM: f64 = 3000.0;
/// Hard limit on raster side length handled by the renderer.
pub const MAX_RASTER_SIDE: u64 = 1_000_000;

/// Converts millimetres at a DPI to a whole number of pixels (rounded).
pub fn mm_to_px(mm: f64, dpi: f64) -> Result<u64> {
    if !mm.is_finite() || !dpi.is_finite() || mm < 0.0 || dpi <= 0.0 {
        bail!("invalid size {mm} mm at {dpi} DPI");
    }
    let px = (mm * dpi / MM_PER_INCH).round();
    if px > MAX_RASTER_SIDE as f64 {
        bail!("{mm} mm at {dpi} DPI is {px} px, above the {MAX_RASTER_SIDE} px limit");
    }
    Ok(px as u64)
}

pub fn px_to_mm(px: u64, dpi: f64) -> f64 {
    px as f64 * MM_PER_INCH / dpi
}

/// Pixel size of an artwork at a given DPI with checked byte arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RasterSpec {
    pub width_mm: f64,
    pub height_mm: f64,
    pub dpi: f64,
    pub width_px: u64,
    pub height_px: u64,
}

impl RasterSpec {
    pub fn new(width_mm: f64, height_mm: f64, dpi: f64) -> Result<Self> {
        validate_artwork_mm(width_mm, height_mm)?;
        let width_px = mm_to_px(width_mm, dpi)?;
        let height_px = mm_to_px(height_mm, dpi)?;
        if width_px == 0 || height_px == 0 {
            bail!("artwork rounds to zero pixels at {dpi} DPI");
        }
        Ok(Self { width_mm, height_mm, dpi, width_px, height_px })
    }

    /// Explicit pixel size (used for previews). DPI is derived from width.
    pub fn with_pixels(width_mm: f64, height_mm: f64, width_px: u64, height_px: u64) -> Result<Self> {
        validate_artwork_mm(width_mm, height_mm)?;
        if width_px == 0 || height_px == 0 || width_px > MAX_RASTER_SIDE || height_px > MAX_RASTER_SIDE {
            bail!("invalid raster size {width_px}x{height_px}");
        }
        Ok(Self { width_mm, height_mm, dpi: width_px as f64 * MM_PER_INCH / width_mm, width_px, height_px })
    }

    pub fn pixel_count(&self) -> Result<u64> {
        self.width_px.checked_mul(self.height_px).ok_or_else(|| anyhow::anyhow!("pixel count overflow"))
    }

    /// Raw RGB8 bytes of the full raster (never allocated; for estimates).
    pub fn rgb_bytes(&self) -> Result<u64> {
        self.pixel_count()?.checked_mul(3).ok_or_else(|| anyhow::anyhow!("byte count overflow"))
    }

    /// Global x pixel coordinate of a millimetre position (exact at both ends).
    pub fn x_px(&self, mm: f64) -> u64 {
        axis_px(mm, self.width_mm, self.width_px)
    }
    pub fn y_px(&self, mm: f64) -> u64 {
        axis_px(mm, self.height_mm, self.height_px)
    }
    /// Millimetres per pixel along each axis.
    pub fn mm_per_px_x(&self) -> f64 {
        self.width_mm / self.width_px as f64
    }
    pub fn mm_per_px_y(&self) -> f64 {
        self.height_mm / self.height_px as f64
    }
}

fn axis_px(mm: f64, total_mm: f64, total_px: u64) -> u64 {
    let v = (mm / total_mm * total_px as f64).round();
    v.clamp(0.0, total_px as f64) as u64
}

pub fn validate_artwork_mm(w: f64, h: f64) -> Result<()> {
    for (name, v) in [("width", w), ("height", h)] {
        if !v.is_finite() || v <= 0.0 {
            bail!("artwork {name} must be positive (got {v})");
        }
        if v > MAX_ARTWORK_MM + 1e-9 {
            bail!("artwork {name} {v} mm exceeds the {MAX_ARTWORK_MM} mm maximum");
        }
    }
    Ok(())
}

/// Pixel edge `i` of `count` equal divisions of `total_px` (integer, exact).
/// Edge 0 is 0 and edge `count` is `total_px`; widths differ by at most 1 px.
pub fn edge(i: u64, count: u64, total_px: u64) -> u64 {
    ((i as u128 * total_px as u128) / count as u128) as u64
}

pub fn format_bytes(b: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1000.0 && u < units.len() - 1 {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 { format!("{b} B") } else { format!("{v:.2} {}", units[u]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_metres_at_300dpi() {
        let r = RasterSpec::new(3000.0, 3000.0, 300.0).unwrap();
        assert_eq!(r.width_px, 35433);
        assert_eq!(r.pixel_count().unwrap(), 35433 * 35433);
        assert_eq!(r.rgb_bytes().unwrap(), 3_766_492_467);
    }

    #[test]
    fn rejects_bad_sizes() {
        assert!(RasterSpec::new(3001.0, 100.0, 300.0).is_err());
        assert!(RasterSpec::new(0.0, 100.0, 300.0).is_err());
        assert!(RasterSpec::new(100.0, f64::NAN, 300.0).is_err());
        assert!(RasterSpec::new(100.0, 100.0, 0.0).is_err());
        assert!(mm_to_px(3000.0, 1e9).is_err());
        assert!(RasterSpec::new(0.01, 0.01, 1.0).is_err()); // rounds to 0 px
    }

    #[test]
    fn mm_px_roundtrip() {
        assert_eq!(mm_to_px(25.4, 300.0).unwrap(), 300);
        assert_eq!(mm_to_px(210.0, 300.0).unwrap(), 2480); // A4 width
        assert!((px_to_mm(300, 300.0) - 25.4).abs() < 1e-9);
    }

    #[test]
    fn edges_are_exact_and_monotonic() {
        let total = 35433;
        let n = 137;
        assert_eq!(edge(0, n, total), 0);
        assert_eq!(edge(n, n, total), total);
        let mut prev = 0;
        for i in 1..=n {
            let e = edge(i, n, total);
            let w = e - prev;
            assert!(w == total / n || w == total / n + 1);
            prev = e;
        }
    }

    #[test]
    fn global_px_mapping_hits_both_ends() {
        let r = RasterSpec::new(1234.5, 800.0, 150.0).unwrap();
        assert_eq!(r.x_px(0.0), 0);
        assert_eq!(r.x_px(1234.5), r.width_px);
        assert_eq!(r.y_px(800.0), r.height_px);
    }
}
