//! Tiled print layout: splits the artwork into print pages.
//!
//! Coordinates are millimetres in the global artwork system. Each page shows
//! an artwork window of at most `usable` size. Neighbouring pages share
//! exactly `overlap` mm of duplicated content (glue/alignment zone); overlap
//! never adds artwork size. Page `i` starts at `i * (usable - overlap)`; the
//! last page in each direction is clipped to the artwork edge (partially
//! filled). After trimming every page except the first in a row/column along
//! its leading crop marks and laying it onto the previous page's alignment
//! line, the assembled artwork is exactly `width × height`.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Paper {
    pub name: &'static str,
    pub width_mm: f64,
    pub height_mm: f64,
}

pub const PAPERS: &[Paper] = &[
    Paper { name: "A4", width_mm: 210.0, height_mm: 297.0 },
    Paper { name: "A3", width_mm: 297.0, height_mm: 420.0 },
    Paper { name: "A2", width_mm: 420.0, height_mm: 594.0 },
    Paper { name: "A1", width_mm: 594.0, height_mm: 841.0 },
    Paper { name: "A0", width_mm: 841.0, height_mm: 1189.0 },
    Paper { name: "Letter", width_mm: 215.9, height_mm: 279.4 },
    Paper { name: "Legal", width_mm: 215.9, height_mm: 355.6 },
    Paper { name: "Tabloid", width_mm: 279.4, height_mm: 431.8 },
];

pub fn paper_by_name(name: &str) -> Option<Paper> {
    PAPERS.iter().copied().find(|p| p.name.eq_ignore_ascii_case(name) || (name.eq_ignore_ascii_case("ledger") && p.name == "Tabloid"))
}

/// PDF viewers and Acrobat limit page sides to 200 in (5080 mm) at UserUnit 1.
pub const MAX_PDF_PAGE_MM: f64 = 5080.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintSettings {
    pub paper_name: String,
    /// Portrait paper size; orientation applied separately.
    pub paper_width_mm: f64,
    pub paper_height_mm: f64,
    pub landscape: bool,
    /// Unprintable printer margins: top, right, bottom, left.
    pub margins_mm: [f64; 4],
    pub overlap_mm: f64,
    pub marks: bool,
    /// Space between the margin and the artwork reserved for crop/alignment marks and labels.
    pub mark_space_mm: f64,
    pub dpi: f64,
    pub jpeg_quality: u8,
}

impl PrintSettings {
    pub fn new(paper: Paper) -> Self {
        Self {
            paper_name: paper.name.to_string(),
            paper_width_mm: paper.width_mm,
            paper_height_mm: paper.height_mm,
            landscape: false,
            margins_mm: [6.0; 4],
            overlap_mm: 10.0,
            marks: true,
            mark_space_mm: 7.0,
            dpi: 300.0,
            jpeg_quality: 92,
        }
    }
    pub fn custom(width_mm: f64, height_mm: f64) -> Self {
        let mut s = Self::new(Paper { name: "Custom", width_mm, height_mm });
        s.paper_name = format!("Custom {width_mm}×{height_mm} mm");
        s
    }
    pub fn sheet_mm(&self) -> (f64, f64) {
        if self.landscape { (self.paper_height_mm, self.paper_width_mm) } else { (self.paper_width_mm, self.paper_height_mm) }
    }
    fn space(&self) -> f64 {
        if self.marks { self.mark_space_mm } else { 0.0 }
    }
    /// Usable artwork area per page after margins and mark space.
    pub fn usable_mm(&self) -> (f64, f64) {
        let (w, h) = self.sheet_mm();
        let [t, r, b, l] = self.margins_mm;
        (w - l - r - 2.0 * self.space(), h - t - b - 2.0 * self.space())
    }
    /// Top-left of the artwork area on the sheet.
    pub fn art_origin_mm(&self) -> (f64, f64) {
        (self.margins_mm[3] + self.space(), self.margins_mm[0] + self.space())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PagePlan {
    pub row: u32,
    pub col: u32,
    pub label: String,
    /// Artwork window in global mm: [x0, x1) × [y0, y1).
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Layout {
    pub cols: u32,
    pub rows: u32,
    pub usable_w: f64,
    pub usable_h: f64,
    pub step_x: f64,
    pub step_y: f64,
    pub art_w: f64,
    pub art_h: f64,
    pub pages: Vec<PagePlan>,
}

pub fn row_label(r: u32) -> String {
    let mut s = String::new();
    let mut n = r + 1;
    while n > 0 {
        let rem = (n - 1) % 26;
        s.insert(0, (b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    s
}

fn split_axis(len: f64, usable: f64, overlap: f64) -> (u32, f64) {
    if len <= usable + 1e-9 {
        return (1, usable - overlap);
    }
    let step = usable - overlap;
    let n = ((len - overlap) / step - 1e-9).ceil().max(1.0) as u32;
    (n, step)
}

pub fn compute_layout(art_w: f64, art_h: f64, s: &PrintSettings) -> Result<Layout> {
    crate::units::validate_artwork_mm(art_w, art_h)?;
    let (sw, sh) = s.sheet_mm();
    if !(sw > 0.0 && sh > 0.0) || sw > MAX_PDF_PAGE_MM || sh > MAX_PDF_PAGE_MM {
        bail!("paper size {sw}×{sh} mm is outside the supported range (0–{MAX_PDF_PAGE_MM} mm)");
    }
    if s.margins_mm.iter().any(|m| !(*m >= 0.0)) || s.overlap_mm < 0.0 || s.mark_space_mm < 0.0 {
        bail!("margins, overlap and mark space must not be negative");
    }
    if !(s.dpi >= 30.0 && s.dpi <= 1200.0) {
        bail!("print resolution must be between 30 and 1200 DPI");
    }
    let (uw, uh) = s.usable_mm();
    if uw <= 0.0 || uh <= 0.0 {
        bail!("margins and marks leave no printable artwork area on {sw}×{sh} mm paper (usable {uw:.1}×{uh:.1} mm)");
    }
    if s.overlap_mm >= uw.min(uh) * 0.5 {
        bail!("overlap {} mm is too large for the usable page area {uw:.1}×{uh:.1} mm (must be under half)", s.overlap_mm);
    }
    let (cols, step_x) = split_axis(art_w, uw, s.overlap_mm);
    let (rows, step_y) = split_axis(art_h, uh, s.overlap_mm);
    if cols as u64 * rows as u64 > 10_000 {
        bail!("{cols}×{rows} pages is too many; choose larger paper");
    }
    let mut pages = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let x0 = c as f64 * step_x;
            let y0 = r as f64 * step_y;
            pages.push(PagePlan {
                row: r,
                col: c,
                label: format!("{}{}", row_label(r), c + 1),
                x0,
                x1: if c + 1 == cols { art_w } else { (x0 + uw).min(art_w) },
                y0,
                y1: if r + 1 == rows { art_h } else { (y0 + uh).min(art_h) },
            });
        }
    }
    Ok(Layout { cols, rows, usable_w: uw, usable_h: uh, step_x, step_y, art_w, art_h, pages })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(p: &str) -> PrintSettings {
        PrintSettings::new(paper_by_name(p).unwrap())
    }

    #[test]
    fn pages_cover_artwork_with_exact_overlap() {
        for (w, h, paper, land) in [(3000.0, 3000.0, "A3", false), (1000.0, 700.0, "A4", true), (500.0, 500.0, "Letter", false), (100.0, 80.0, "A4", false)] {
            let mut s = settings(paper);
            s.landscape = land;
            let l = compute_layout(w, h, &s).unwrap();
            assert!(l.pages.iter().all(|p| p.x1 - p.x0 <= l.usable_w + 1e-9 && p.y1 - p.y0 <= l.usable_h + 1e-9));
            let first_row: Vec<&PagePlan> = l.pages.iter().filter(|p| p.row == 0).collect();
            assert_eq!(first_row[0].x0, 0.0);
            assert_eq!(first_row.last().unwrap().x1, w);
            for pair in first_row.windows(2) {
                // Adjacent pages overlap by exactly the overlap: no gaps, no double counting.
                assert!((pair[0].x1 - pair[1].x0 - s.overlap_mm).abs() < 1e-9, "{w} {paper}");
            }
            // Assembled (trimmed) width equals the artwork width exactly.
            let trimmed: f64 = first_row.iter().enumerate().map(|(i, p)| (p.x1 - p.x0) - if i > 0 { s.overlap_mm } else { 0.0 }).sum();
            assert!((trimmed - w).abs() < 1e-9);
            let first_col: Vec<&PagePlan> = l.pages.iter().filter(|p| p.col == 0).collect();
            let trimmed_h: f64 = first_col.iter().enumerate().map(|(i, p)| (p.y1 - p.y0) - if i > 0 { s.overlap_mm } else { 0.0 }).sum();
            assert!((trimmed_h - h).abs() < 1e-9);
        }
    }

    #[test]
    fn page_count_3m_on_a3() {
        let s = settings("A3");
        let l = compute_layout(3000.0, 3000.0, &s).unwrap();
        // A3 portrait: usable 297-12-14=271 wide, 420-12-14=394 high; step 261 / 384.
        assert_eq!((l.usable_w, l.usable_h), (271.0, 394.0));
        assert_eq!(l.cols, ((3000.0f64 - 10.0) / 261.0).ceil() as u32);
        assert_eq!(l.rows, ((3000.0f64 - 10.0) / 384.0).ceil() as u32);
    }

    #[test]
    fn single_page_and_partial_edges() {
        let s = settings("A4");
        let l = compute_layout(150.0, 200.0, &s).unwrap();
        assert_eq!(l.pages.len(), 1);
        assert_eq!((l.pages[0].x1, l.pages[0].y1), (150.0, 200.0));
        let l = compute_layout(300.0, 200.0, &s).unwrap();
        assert_eq!(l.cols, 2);
        assert!(l.pages[1].x1 - l.pages[1].x0 < l.usable_w); // partially filled
    }

    #[test]
    fn rejects_no_usable_area() {
        let mut s = settings("A4");
        s.margins_mm = [100.0, 100.0, 100.0, 100.0];
        assert!(compute_layout(500.0, 500.0, &s).unwrap_err().to_string().contains("no printable"));
        let mut s = settings("A4");
        s.overlap_mm = 150.0;
        assert!(compute_layout(500.0, 500.0, &s).is_err());
    }

    #[test]
    fn labels() {
        assert_eq!(row_label(0), "A");
        assert_eq!(row_label(25), "Z");
        assert_eq!(row_label(26), "AA");
    }
}
