//! Offline Mosaic Story Card and Palette Coach diagnostics.
//!
//! Computes human-readable gift placard text from [`OutputStats`] and analyses
//! per-tile CIE Lab matching errors against the photo pool to pinpoint which
//! colour/brightness ranges would most improve the mosaic. Also builds a
//! privacy-preserving stats-only prompt for optional LLM polishing.

use crate::catalog::{unhex, Catalog};
use crate::color::{de2, Lab};
use crate::descriptor::{crop_descriptor, mean_lab, CELLS};
use crate::photo_index::friendly_date;
use crate::project::{OutputStats, Project};
use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct StoryCard {
    pub headline: String,
    pub story: String,
    pub palette_coach: Vec<String>,
}

const BUCKET_NAMES: [&str; 8] = [
    "deep shadows (very dark scenes or night shots)",
    "bright highlights (snow, clouds or bright skies)",
    "neutral midtones (stone, concrete, monochrome or overcast scenes)",
    "warm reds & pinks (sunsets, bricks, flowers)",
    "oranges & golden yellows (golden hour, autumn leaves, sand)",
    "greens & foliage (forests, grass, plants)",
    "cyans & blues (clear skies, ocean, water)",
    "purples & magentas (twilight, neon, florals)",
];

fn color_bucket(lab: Lab) -> usize {
    let (l, a, b) = (lab[0], lab[1], lab[2]);
    let chroma = (a * a + b * b).sqrt();
    if l < 24.0 {
        return 0;
    }
    if l > 82.0 && chroma < 18.0 {
        return 1;
    }
    if chroma < 11.0 {
        return 2;
    }
    let hue = b.atan2(a).to_degrees().rem_euclid(360.0);
    if !(45.0..345.0).contains(&hue) {
        3 // red / pink
    } else if hue < 108.0 {
        4 // orange / yellow
    } else if hue < 180.0 {
        5 // green
    } else if hue < 310.0 {
        6 // cyan / blue (sky, cobalt, pure sRGB blue span ~180..307° in CIE Lab)
    } else {
        7 // purple / magenta
    }
}

/// Analyse per-tile colour mismatches and produce actionable Palette Coach advice.
pub fn analyze_palette(cat: &Catalog, p: &Project) -> Result<Vec<String>> {
    let cells = p.target_cells()?;
    let gw = p.grid.cols as usize * CELLS;
    let cols = p.grid.cols as usize;
    let source_grids: Vec<Option<Vec<u8>>> = p
        .sources
        .iter()
        .map(|s| unhex(&s.hash).ok().and_then(|h| cat.descriptor(&h).ok().flatten()).map(|r| r.grid))
        .collect();

    let mut bucket_tiles = [0usize; 8];
    let mut bucket_err_sum = [0.0f64; 8];
    let mut bucket_high_err = [0usize; 8];
    let mut total_placed = 0usize;

    for (i, t) in p.tiles.iter().enumerate() {
        if t.is_empty() {
            continue;
        }
        let Some(grid_bytes) = source_grids.get(t.photo as usize).and_then(|g| g.as_ref()) else {
            continue;
        };
        let (tx, ty) = (i % cols, i / cols);
        let mut tsum = [0.0f32; 3];
        for cy in 0..CELLS {
            for cx in 0..CELLS {
                let c = cells[(ty * CELLS + cy) * gw + (tx * CELLS + cx)];
                tsum[0] += c[0];
                tsum[1] += c[1];
                tsum[2] += c[2];
            }
        }
        let inv = 1.0 / (CELLS * CELLS) as f32;
        let tmean = [tsum[0] * inv, tsum[1] * inv, tsum[2] * inv];
        let pmean = mean_lab(&crop_descriptor(grid_bytes, t.crop));
        let err = de2(&tmean, &pmean).sqrt() as f64;
        let b = color_bucket(tmean);
        bucket_tiles[b] += 1;
        bucket_err_sum[b] += err;
        if err > 12.0 {
            bucket_high_err[b] += 1;
        }
        total_placed += 1;
    }

    let mut tips = Vec::new();
    if total_placed == 0 {
        return Ok(tips);
    }

    let mut gaps: Vec<(usize, f64, usize)> = (0..8)
        .filter(|&b| bucket_tiles[b] >= 3)
        .map(|b| (b, bucket_err_sum[b] / bucket_tiles[b] as f64, bucket_high_err[b]))
        .filter(|&(_, mean_de, high)| mean_de >= 11.0 && high > 0)
        .collect();
    gaps.sort_by(|a, b| (b.1 * (b.2 as f64).sqrt()).total_cmp(&(a.1 * (a.2 as f64).sqrt())));

    if gaps.is_empty() {
        tips.push(format!(
            "Strong colour coverage: average tile error is {:.1} ΔE with no major missing colour families.",
            p.stats.mean_delta_e
        ));
    } else {
        for &(b, mean_de, high) in gaps.iter().take(3) {
            let share = bucket_tiles[b] as f64 * 100.0 / total_placed as f64;
            tips.push(format!(
                "Colour gap in {} — {:.0}% of tiles (avg ΔE {:.1}, {} tiles > 12 ΔE). Adding more photos in this tone will sharpen the likeness.",
                BUCKET_NAMES[b], share, mean_de, high
            ));
        }
    }

    if p.stats.nearby_repeats > total_placed / 15 {
        tips.push(format!(
            "{} adjacent tiles repeat the same photo. Raising Avoid nearby repeats or adding more photos in the dominant colours will reduce clusters.",
            p.stats.nearby_repeats
        ));
    }
    if p.stats.mean_delta_e > 16.0 && p.settings.color_correction < 0.05 {
        tips.push("Tip: Average colour error is above 16 ΔE. Try enabling 10–20% Colour correction in Step 3 Advanced for a closer overall likeness.".into());
    }
    Ok(tips)
}

/// Build an offline gift-ready Story Card and Palette Coach summary.
pub fn build_story_card(cat: &Catalog, p: &Project, st: &OutputStats) -> Result<StoryCard> {
    let coach = analyze_palette(cat, p)?;
    Ok(offline_story(p, st, coach))
}

pub fn offline_story(p: &Project, st: &OutputStats, palette_coach: Vec<String>) -> StoryCard {
    let (w_cm, h_cm) = (st.artwork_mm.0 / 10.0, st.artwork_mm.1 / 10.0);
    let span = match (&st.exif.date_first, &st.exif.date_last) {
        (Some(a), Some(b)) if a.get(..4) == b.get(..4) => a.get(..4).map(|y| format!("captured in {y}")),
        (Some(a), Some(b)) => Some(format!("spanning {} to {}", friendly_date(a), friendly_date(b))),
        _ => None,
    };
    let headline = match &span {
        Some(sp) => format!("{} tiles · {} photos {sp}", st.tiles, st.distinct_photos_used),
        None => format!("{} tiles crafted from {} distinct photos", st.tiles, st.distinct_photos_used),
    };
    let mut sentences = Vec::new();
    sentences.push(format!(
        "This {:.0} × {:.0} cm photo mosaic brings together {} distinct photographs across {} tiles ({} × {}).",
        w_cm, h_cm, st.distinct_photos_used, st.tiles, st.grid.0, st.grid.1
    ));
    if let Some(sp) = &span {
        sentences.push(format!("Its moments were {sp}."));
    }
    if let Some(cam) = st.exif.cameras.first() {
        let lens_part = st
            .exif
            .lenses
            .first()
            .map(|l| format!(", most often paired with the {}", l.value))
            .unwrap_or_default();
        sentences.push(format!(
            "The collection's favourite camera was the {} ({} photos){lens_part}.",
            cam.value, cam.photos
        ));
    }
    let (tw, th) = p.grid.tile_mm();
    let dist = (tw.max(th) * 0.15).max(1.0);
    sentences.push(format!(
        "Up close, every {:.0} × {:.0} mm tile is a complete memory ({} photos appear only once); step back about {:.0} m to see the whole picture come together.",
        tw, th, st.photos_used_once, dist
    ));
    StoryCard {
        headline,
        story: sentences.join(" "),
        palette_coach,
    }
}

/// Privacy-preserving stats-only prompt for optional LLM story & coaching polish.
/// Contains zero file paths, zero filenames, and zero image bytes.
pub fn ai_story_prompt(card: &StoryCard, st: &OutputStats) -> String {
    let cameras: Vec<String> = st.exif.cameras.iter().take(3).map(|c| format!("{} ({})", c.value, c.photos)).collect();
    let lenses: Vec<String> = st.exif.lenses.iter().take(3).map(|l| format!("{} ({})", l.value, l.photos)).collect();
    format!(
        "Artwork size: {:.0} x {:.0} cm ({} x {} grid, {} tiles).\n\
         Distinct photos used: {} ({} used once, max uses {}).\n\
         Mean colour error: {:.1} ΔE.\n\
         EXIF date span: {} to {}.\n\
         Top cameras: {}.\n\
         Top lenses: {}.\n\
         Palette diagnostics:\n- {}\n\n\
         Write two short sections:\n\
         1) STORY CARD: A warm 3–4 sentence gallery/gift placard about this mosaic.\n\
         2) PALETTE COACH: 2–3 crisp, practical photography tips based on the diagnostics above.",
        st.artwork_mm.0 / 10.0,
        st.artwork_mm.1 / 10.0,
        st.grid.0,
        st.grid.1,
        st.tiles,
        st.distinct_photos_used,
        st.photos_used_once,
        st.max_uses_of_one_photo,
        st.mean_color_error_delta_e,
        st.exif.date_first.as_deref().unwrap_or("unknown"),
        st.exif.date_last.as_deref().unwrap_or("unknown"),
        if cameras.is_empty() { "not recorded".into() } else { cameras.join(", ") },
        if lenses.is_empty() { "not recorded".into() } else { lenses.join(", ") },
        if card.palette_coach.is_empty() { "No major gaps".into() } else { card.palette_coach.join("\n- ") }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_buckets_separate_tones_and_hues() {
        assert_eq!(color_bucket([10.0, 0.0, 0.0]), 0); // deep shadow
        assert_eq!(color_bucket([92.0, 1.0, -2.0]), 1); // bright highlight
        assert_eq!(color_bucket([55.0, 2.0, 1.0]), 2); // neutral midtone
        assert_eq!(color_bucket([50.0, 60.0, 20.0]), 3); // warm red
        assert_eq!(color_bucket([65.0, -45.0, 40.0]), 5); // green
        assert_eq!(color_bucket([50.0, -5.0, -45.0]), 6); // blue
    }
}
