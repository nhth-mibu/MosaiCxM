//! How sharp the tiles of a print look from a given distance.
//!
//! A person with normal sight sees detail down to about one minute of arc.
//! From `d` metres that is about `0.29 · d` mm, so a print looks sharp from
//! `d` when it has about `87 / d` pixels per inch of real photo detail.
//! A tile shows a fixed number of photo pixels. On a big tile, or with a small
//! photo, there are fewer pixels per inch and the tile looks soft up close,
//! whatever DPI the file is exported at.

use crate::project::Project;
use crate::units::MM_PER_INCH;

/// Smallest detail a person with normal sight can see, in minutes of arc.
pub const ACUITY_ARCMIN: f64 = 1.0;

/// Pixels per inch a print needs to look sharp from `distance_m`.
pub fn needed_dpi(distance_m: f64) -> f64 {
    MM_PER_INCH / (distance_m * 1000.0 * (ACUITY_ARCMIN / 60.0).to_radians().tan())
}

/// Closest distance (m) from which a print with `dpi` pixels per inch looks sharp.
pub fn sharp_from_m(dpi: f64) -> f64 {
    if dpi > 0.0 {
        needed_dpi(1.0) / dpi
    } else {
        f64::INFINITY
    }
}

/// Real photo detail of each tile on the print, in pixels per inch.
/// `None` for tiles outside the mosaic outline.
pub fn tile_dpi(p: &Project) -> Vec<Option<f64>> {
    let (tw, th) = p.grid.tile_mm();
    p.tiles
        .iter()
        .map(|t| {
            if t.is_empty() {
                return None;
            }
            let s = p.sources.get(t.photo as usize)?;
            let px_w = t.crop[2] as f64 * s.width as f64;
            let px_h = t.crop[3] as f64 * s.height as f64;
            Some((px_w / tw).min(px_h / th) * MM_PER_INCH)
        })
        .collect()
}

/// Sharpness of all tiles of a project.
#[derive(Debug, Clone)]
pub struct Sharpness {
    /// Tiles with a photo.
    pub tiles: usize,
    /// For each tile with a photo, the closest distance (m) it looks sharp from, largest first.
    sharp_from: Vec<f64>,
    /// DPI of the print (infinite when not known).
    print_dpi: f64,
}

impl Sharpness {
    /// Sharpness when printed at `print_dpi` (no tile can have more detail
    /// than the print). `None` looks at the photos only.
    pub fn of(p: &Project, print_dpi: Option<f64>) -> Self {
        let cap = print_dpi.filter(|d| *d > 0.0).unwrap_or(f64::INFINITY);
        let dpi: Vec<f64> = tile_dpi(p).into_iter().flatten().collect();
        let mut sharp_from: Vec<f64> = dpi.into_iter().map(|d| sharp_from_m(d.min(cap))).collect();
        sharp_from.sort_unstable_by(|a, b| b.total_cmp(a));
        Self { tiles: sharp_from.len(), sharp_from, print_dpi: cap }
    }

    /// Number of tiles that look soft from `distance_m`.
    pub fn soft_at(&self, distance_m: f64) -> usize {
        self.sharp_from.partition_point(|&d| d > distance_m)
    }

    /// Closest distance (m) from which every tile looks sharp.
    pub fn all_sharp_from_m(&self) -> f64 {
        self.sharp_from.first().copied().unwrap_or(0.0)
    }

    /// A warning when some tiles look soft from `near_m`, else `None`.
    pub fn warning(&self, near_m: f64) -> Option<String> {
        let n = self.soft_at(near_m);
        if n == 0 {
            return None;
        }
        let from = self.all_sharp_from_m();
        if self.print_dpi < needed_dpi(near_m) {
            return Some(format!("The DPI is too low to look sharp from closer than {from:.1} m. Use a DPI of {:.0} or more.", needed_dpi(near_m)));
        }
        let tiles = if n == 1 { "1 tile uses a photo".to_string() } else { format!("{n} tiles use photos") };
        Some(format!("{tiles} with too few pixels for the tile size. They look soft from closer than {from:.1} m. To fix this, make the tiles smaller."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acuity_numbers() {
        // About 87 DPI from 1 m, 29 DPI from 3 m.
        assert!((needed_dpi(1.0) - 87.3).abs() < 0.1, "{}", needed_dpi(1.0));
        assert!((needed_dpi(3.0) - 29.1).abs() < 0.1);
        assert!((sharp_from_m(needed_dpi(2.5)) - 2.5).abs() < 1e-9);
        assert_eq!(sharp_from_m(0.0), f64::INFINITY);
    }

    #[test]
    fn counts_soft_tiles() {
        let s = Sharpness { tiles: 4, sharp_from: vec![4.0, 2.0, 0.5, 0.1], print_dpi: f64::INFINITY };
        assert_eq!(s.soft_at(1.0), 2);
        assert_eq!(s.soft_at(3.0), 1);
        assert_eq!(s.soft_at(5.0), 0);
        assert_eq!(s.all_sharp_from_m(), 4.0);
        assert!(s.warning(1.0).unwrap().starts_with("2 tiles use photos"));
        assert!(s.warning(5.0).is_none());
    }
}
