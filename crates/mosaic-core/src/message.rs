//! Hidden message: a word, name or date written into the mosaic with tile
//! brightness.
//!
//! The text is drawn into the target motif before matching: tiles under the
//! letters ask for lighter (or darker) photos. Up close every tile is still a
//! normal photo; from a distance the brightness pattern reads as text.
//!
//! The text is set in Liberation Sans Bold (embedded, SIL OFL 1.1, see
//! `fonts/`), centred and scaled to fit the artwork. Type `|` or a line break
//! to start a new line.

use crate::descriptor::CELLS;
use crate::matching::{GridSpec, Target};
use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

static FONT: &[u8] = include_bytes!("../fonts/LiberationSans-Bold.ttf");

/// Letters smaller than this many tiles cannot be read.
pub const MIN_LETTER_TILES: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tone {
    /// Light letters on a dark picture, dark letters on a light one.
    #[default]
    Auto,
    Light,
    Dark,
}

impl Tone {
    pub const ALL: [Tone; 3] = [Tone::Auto, Tone::Light, Tone::Dark];
    pub fn parse(s: &str) -> Result<Tone> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Tone::Auto),
            "light" | "bright" => Ok(Tone::Light),
            "dark" => Ok(Tone::Dark),
            _ => bail!("unknown message tone '{s}' (auto, light, dark)"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiddenMessage {
    pub text: String,
    /// 0–1: how far letter tiles move toward white or black.
    pub strength: f32,
    #[serde(default)]
    pub tone: Tone,
}

impl HiddenMessage {
    pub const DEFAULT_STRENGTH: f32 = 0.5;

    pub fn new(text: &str) -> Self {
        Self { text: text.to_string(), strength: Self::DEFAULT_STRENGTH, tone: Tone::Auto }
    }

    pub fn lines(&self) -> Vec<&str> {
        self.text.split(['\n', '|']).map(str::trim).collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.lines().iter().all(|l| l.is_empty()) {
            bail!("the hidden message is empty");
        }
        if !(self.strength > 0.0 && self.strength <= 1.0) {
            bail!("message strength must be above 0 and at most 1");
        }
        let font = font();
        for c in self.text.chars() {
            if !c.is_whitespace() && c != '|' && font.glyph_id(c).0 == 0 {
                bail!("the hidden message cannot use '{c}': the built-in font does not have it");
            }
        }
        Ok(())
    }
}

fn font() -> FontRef<'static> {
    FontRef::try_from_slice(FONT).expect("embedded font is valid")
}

/// Letter coverage (0–1) per target cell, row-major, `cols·CELLS × rows·CELLS`.
#[derive(Debug, Clone)]
pub struct MessageMask {
    pub width: usize,
    pub height: usize,
    pub cover: Vec<f32>,
    /// Height of a capital letter, in tiles.
    pub letter_tiles: f32,
}

/// Reference size used to measure the text before it is scaled to the grid.
const REF: f32 = 100.0;

struct Layout {
    /// Glyph ids and baseline origins at the reference size.
    glyphs: Vec<(ab_glyph::GlyphId, f32, f32)>,
    /// Ink centre at the reference size.
    centre: (f32, f32),
    mm_per_ref: f32,
    letter_tiles: f32,
}

fn layout(font: &FontRef<'static>, msg: &HiddenMessage, grid: &GridSpec) -> Result<Layout> {
    msg.validate()?;
    let lines = msg.lines();
    let scaled = font.as_scaled(PxScale::from(REF));
    let line_h = scaled.height() + scaled.line_gap();

    let mut glyphs = Vec::new();
    let mut ink = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for (li, line) in lines.iter().enumerate() {
        let mut x = 0.0;
        let mut prev = None;
        let mut row = Vec::new();
        for c in line.chars() {
            let id = font.glyph_id(c);
            if let Some(p) = prev {
                x += scaled.kern(p, id);
            }
            row.push((id, x));
            x += scaled.h_advance(id);
            prev = Some(id);
        }
        let shift = -x / 2.0; // centre each line
        let base = scaled.ascent() + li as f32 * line_h;
        for (id, gx) in row {
            let gx = gx + shift;
            if let Some(o) = font.outline_glyph(id.with_scale_and_position(REF, point(gx, base))) {
                let b = o.px_bounds();
                ink = [ink[0].min(b.min.x), ink[1].min(b.min.y), ink[2].max(b.max.x), ink[3].max(b.max.y)];
            }
            glyphs.push((id, gx, base));
        }
    }
    if ink[0] >= ink[2] {
        bail!("the hidden message has no visible letters");
    }
    let (ink_w, ink_h) = (ink[2] - ink[0], ink[3] - ink[1]);

    // Fit inside the artwork (mm): wide, but not taller than about half of it.
    let (w_mm, h_mm) = (grid.width_mm as f32, grid.height_mm as f32);
    let n = lines.len() as f32;
    let max_h = h_mm * (0.42 + 0.14 * (n - 1.0)).min(0.8);
    let mm_per_ref = (0.86 * w_mm / ink_w).min(max_h / ink_h);

    let cap = font.outline_glyph(font.glyph_id('H').with_scale(REF)).map(|o| o.px_bounds().height()).unwrap_or(REF * 0.7);
    let letter_tiles = cap * mm_per_ref / (h_mm / grid.rows as f32);
    Ok(Layout { glyphs, centre: ((ink[0] + ink[2]) / 2.0, (ink[1] + ink[3]) / 2.0), mm_per_ref, letter_tiles })
}

/// Height of a capital letter in tiles, without drawing the message.
pub fn letter_tiles(msg: &HiddenMessage, grid: &GridSpec) -> Result<f32> {
    Ok(layout(&font(), msg, grid)?.letter_tiles)
}

/// Draw the message into a cell mask that fits the artwork.
pub fn mask(msg: &HiddenMessage, grid: &GridSpec) -> Result<MessageMask> {
    let font = font();
    let l = layout(&font, msg, grid)?;
    let (gw, gh) = (grid.cols as usize * CELLS, grid.rows as usize * CELLS);
    let (sx, sy) = (l.mm_per_ref * gw as f32 / grid.width_mm as f32, l.mm_per_ref * gh as f32 / grid.height_mm as f32);
    let scale = PxScale { x: REF * sx, y: REF * sy };
    let (cx, cy) = l.centre;

    let mut cover = vec![0f32; gw * gh];
    for (id, gx, gy) in l.glyphs {
        let pos = point((gx - cx) * sx + gw as f32 / 2.0, (gy - cy) * sy + gh as f32 / 2.0);
        let Some(o) = font.outline_glyph(id.with_scale_and_position(scale, pos)) else { continue };
        let b = o.px_bounds();
        o.draw(|x, y, c| {
            let (px, py) = (b.min.x as i64 + x as i64, b.min.y as i64 + y as i64);
            if px >= 0 && py >= 0 && (px as usize) < gw && (py as usize) < gh {
                let v = &mut cover[py as usize * gw + px as usize];
                *v = v.max(c.min(1.0));
            }
        });
    }
    Ok(MessageMask { width: gw, height: gh, cover, letter_tiles: l.letter_tiles })
}

/// Push the target lighter or darker under the letters.
/// Returns the letter height in tiles.
pub fn apply(target: &mut Target, grid: &GridSpec, msg: &HiddenMessage) -> Result<f32> {
    let tiles = letter_tiles(msg, grid)?;
    if tiles < MIN_LETTER_TILES {
        bail!(
            "the hidden message is too long for this mosaic: its letters would be only {:.1} tiles tall. Use fewer letters, more lines or smaller tiles",
            tiles
        );
    }
    let m = mask(msg, grid)?;
    let mut cells = target.cells();
    if cells.len() != m.cover.len() {
        bail!("hidden message does not match the grid");
    }
    let light = match msg.tone {
        Tone::Light => true,
        Tone::Dark => false,
        Tone::Auto => {
            let (mut sum, mut w) = (0.0f64, 0.0f64);
            for (c, &k) in cells.iter().zip(&m.cover) {
                sum += c[0] as f64 * k as f64;
                w += k as f64;
            }
            w == 0.0 || sum / w < 55.0
        }
    };
    let k = 0.6 * msg.strength;
    for (c, &cov) in cells.iter_mut().zip(&m.cover) {
        if cov <= 0.0 {
            continue;
        }
        let t = k * cov;
        c[0] = if light { c[0] + (100.0 - c[0]) * t } else { c[0] * (1.0 - t) };
    }
    *target = Target::from_cells(target.cols, target.rows, &cells);
    Ok(m.letter_tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(cols: u32, rows: u32) -> GridSpec {
        GridSpec::new(cols as f64 * 10.0, rows as f64 * 10.0, cols, rows).unwrap()
    }

    #[test]
    fn mask_is_centred_and_sized() {
        let m = mask(&HiddenMessage::new("LOVE"), &grid(80, 60)).unwrap();
        let total: f32 = m.cover.iter().sum();
        assert!(total > 0.0);
        // Centre of mass near the middle.
        let (mut x, mut y) = (0.0, 0.0);
        for (i, &c) in m.cover.iter().enumerate() {
            x += (i % m.width) as f32 * c;
            y += (i / m.width) as f32 * c;
        }
        let (x, y) = (x / total / m.width as f32, y / total / m.height as f32);
        assert!((x - 0.5).abs() < 0.05 && (y - 0.5).abs() < 0.08, "{x} {y}");
        assert!(m.letter_tiles > 10.0, "{}", m.letter_tiles);
        // Borders stay clear.
        assert!(m.cover[..m.width].iter().all(|&c| c == 0.0));
    }

    #[test]
    fn lines_and_letters() {
        let one = mask(&HiddenMessage::new("ANNA & TOM 2026"), &grid(80, 60)).unwrap();
        let two = mask(&HiddenMessage::new("ANNA & TOM | 2026"), &grid(80, 60)).unwrap();
        assert!(two.letter_tiles > one.letter_tiles);
        assert!(HiddenMessage::new("Nhật Đức").validate().is_ok());
        assert!(HiddenMessage::new("  | ").validate().is_err());
        let mut m = HiddenMessage::new("HI");
        m.strength = 0.0;
        assert!(m.validate().is_err());
    }

    #[test]
    fn apply_moves_brightness() {
        let g = grid(40, 30);
        let cells = vec![[50.0f32, 10.0, -5.0]; g.cols as usize * CELLS * g.rows as usize * CELLS];
        let base = Target::from_cells(g.cols, g.rows, &cells);
        let mut t = base.clone();
        let msg = HiddenMessage::new("HI");
        let tiles = apply(&mut t, &g, &msg).unwrap();
        assert!(tiles >= MIN_LETTER_TILES);
        let m = mask(&msg, &g).unwrap();
        let after = t.cells();
        for ((a, &c), b) in after.iter().zip(&m.cover).zip(&cells) {
            if c > 0.99 {
                assert!(a[0] > 60.0, "letters get lighter on a mid-grey picture");
            } else if c == 0.0 {
                assert_eq!(a[0], b[0]);
            }
            assert_eq!(a[1], b[1]);
        }
        let mut dark = msg.clone();
        dark.tone = Tone::Dark;
        let mut t2 = base.clone();
        apply(&mut t2, &g, &dark).unwrap();
        assert!(t2.cells().iter().zip(&m.cover).any(|(a, &c)| c > 0.99 && a[0] < 40.0));
    }

    #[test]
    fn too_small_is_an_error() {
        let g = grid(12, 8);
        let cells = vec![[50.0f32, 0.0, 0.0]; g.cols as usize * CELLS * g.rows as usize * CELLS];
        let mut t = Target::from_cells(g.cols, g.rows, &cells);
        assert!(apply(&mut t, &g, &HiddenMessage::new("HAPPY BIRTHDAY GRANDMA")).is_err());
    }
}
