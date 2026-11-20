//! "Find my photo": where each source photo sits on the mosaic, and a
//! printable index (PDF) so people can find their photos on the big print.
//!
//! The print is split into lettered areas, like a map: rows of areas have
//! letters (A at the top) and columns of areas have numbers (1 at the left),
//! the same way the tiled PDF names its pages. Each tile also gets its
//! distance from the left and top edges, so it can be found with a tape.

use crate::atomic::TempOutput;
use crate::catalog::{unhex, Catalog, Location};
use crate::job::Job;
use crate::layout::row_label;
use crate::pdf::{Content, PdfOut, PT_PER_MM};
use crate::project::Project;
use anyhow::Result;
use image::RgbImage;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// About this many areas along the long side of the mosaic.
const AREAS_LONG_SIDE: u32 = 10;

/// How the mosaic is split into areas for finding tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Areas {
    /// Tiles per area along each side.
    pub tiles: u32,
    pub across: u32,
    pub down: u32,
}

impl Areas {
    pub fn for_grid(cols: u32, rows: u32) -> Self {
        let tiles = cols.max(rows).max(1).div_ceil(AREAS_LONG_SIDE).max(1);
        Self { tiles, across: cols.max(1).div_ceil(tiles), down: rows.max(1).div_ceil(tiles) }
    }
    /// Area name of the tile at (row, col), such as "C4".
    pub fn label(&self, row: u32, col: u32) -> String {
        format!("{}{}", row_label(row / self.tiles), col / self.tiles + 1)
    }
}

/// Where one tile is on the print.
#[derive(Debug, Clone, PartialEq)]
pub struct TilePlace {
    /// 1-based, from the top.
    pub row: u32,
    /// 1-based, from the left.
    pub col: u32,
    pub area: String,
    /// Centre of the tile, from the left and top edges of the artwork.
    pub x_mm: f64,
    pub y_mm: f64,
    /// Longest side of the mosaic, which sets the unit (mm, cm or m).
    pub scale_mm: f64,
}

impl TilePlace {
    pub fn of(p: &Project, tile: usize) -> Self {
        let cols = p.grid.cols.max(1);
        let (r, c) = ((tile as u32) / cols, (tile as u32) % cols);
        let (tw, th) = p.grid.tile_mm();
        let areas = Areas::for_grid(p.grid.cols, p.grid.rows);
        Self { row: r + 1, col: c + 1, area: areas.label(r, c), x_mm: (c as f64 + 0.5) * tw, y_mm: (r as f64 + 0.5) * th, scale_mm: p.grid.width_mm.max(p.grid.height_mm) }
    }
    /// "Area C4, row 23, column 47"
    pub fn short(&self) -> String {
        format!("Area {}, row {}, column {}", self.area, self.row, self.col)
    }
    /// "41 cm from the left, 63 cm from the top"
    pub fn distance(&self) -> String {
        format!("{} from the left, {} from the top", length_in(self.x_mm, self.scale_mm), length_in(self.y_mm, self.scale_mm))
    }
}

fn length(mm: f64) -> String {
    length_in(mm, mm)
}

/// `mm` in the unit that suits a length of `scale` mm, so all distances on
/// one mosaic use the same unit.
fn length_in(mm: f64, scale: f64) -> String {
    if scale < 300.0 { format!("{mm:.0} mm") } else if scale < 10_000.0 { format!("{:.0} cm", mm / 10.0) } else { format!("{:.2} m", mm / 1000.0) }
}

/// Makes an EXIF date ("2023:07:14 15:02:11") easy to read ("14 Jul 2023, 15:02").
/// Other text is returned as it is.
pub fn friendly_date(taken: &str) -> String {
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let t = taken.trim();
    let num = |a: usize, b: usize| t.get(a..b).and_then(|s| s.parse::<u32>().ok());
    match (num(0, 4), num(5, 7), num(8, 10)) {
        (Some(y), Some(m @ 1..=12), Some(d @ 1..=31)) if y > 0 => {
            let date = format!("{d} {} {y}", MONTHS[m as usize - 1]);
            match (num(11, 13), num(14, 16)) {
                (Some(h), Some(mi)) if (h, mi) != (0, 0) => format!("{date}, {h:02}:{mi:02}"),
                _ => date,
            }
        }
        _ => t.to_string(),
    }
}

/// One photo of the mosaic and every tile that shows it.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Index into `Project::sources`.
    pub source: usize,
    /// First known location in the catalogue (the file may have moved since).
    pub location: Option<Location>,
    /// Tiles (row-major) that show this photo.
    pub tiles: Vec<usize>,
}

impl IndexEntry {
    pub fn name(&self) -> String {
        match &self.location {
            Some(l) => l.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| l.path.display().to_string()),
            None => format!("Photo {}", self.source + 1),
        }
    }
    pub fn path(&self) -> Option<&PathBuf> {
        self.location.as_ref().map(|l| &l.path)
    }
    pub fn taken(&self) -> Option<&str> {
        self.location.as_ref().and_then(|l| l.exif.taken.as_deref()).filter(|s| !s.trim().is_empty())
    }
}

/// Tiles that show each source photo (indexed like `Project::sources`).
pub fn tiles_by_source(p: &Project) -> Vec<Vec<usize>> {
    let mut v = vec![Vec::new(); p.sources.len()];
    for (i, t) in p.tiles.iter().enumerate() {
        if !t.is_empty() {
            if let Some(list) = v.get_mut(t.photo as usize) {
                list.push(i);
            }
        }
    }
    v
}

/// Every photo used, sorted by date taken (photos without a date last), then by name.
pub fn build_index(cat: &Catalog, p: &Project, job: &Job) -> Result<Vec<IndexEntry>> {
    let libs = p.library_ids();
    let tiles = tiles_by_source(p);
    let mut out = Vec::with_capacity(p.sources.len());
    for (i, (s, t)) in p.sources.iter().zip(tiles).enumerate() {
        if i % 256 == 0 {
            job.check()?;
            job.report("listing photos", i as u64, p.sources.len() as u64);
        }
        if t.is_empty() {
            continue;
        }
        let location = cat.locations(&unhex(&s.hash)?, &libs)?.into_iter().next();
        out.push(IndexEntry { source: i, location, tiles: t });
    }
    out.sort_by_cached_key(|e| (e.taken().is_none(), e.taken().map(str::to_string), e.name().to_lowercase()));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Printable index (PDF).

const SHEET_W: f64 = 210.0;
const SHEET_H: f64 = 297.0;
const MARGIN: f64 = 12.0;
const ROW_H: f64 = 23.5;
const THUMB: f64 = 17.0;
const COL_GAP: f64 = 6.0;
/// Positions listed per photo; the rest are counted.
const MAX_PLACES: usize = 3;

/// Rough Helvetica text width in mm (average glyph ≈ 0.52 em).
fn text_w(s: &str, size_pt: f64) -> f64 {
    s.chars().count() as f64 * size_pt * 0.52 / PT_PER_MM
}

/// Cuts `s` to fit `w` mm, from the end (or from the start for paths).
fn fit(s: &str, size_pt: f64, w: f64, keep_end: bool) -> String {
    if text_w(s, size_pt) <= w {
        return s.to_string();
    }
    let n = ((w / (size_pt * 0.52 / PT_PER_MM)) as usize).saturating_sub(3).max(1);
    let chars: Vec<char> = s.chars().collect();
    if keep_end {
        format!("...{}", chars[chars.len().saturating_sub(n)..].iter().collect::<String>())
    } else {
        format!("{}...", chars[..n.min(chars.len())].iter().collect::<String>())
    }
}

/// Width and height of a JPEG, read from its header.
fn jpeg_size(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Jpeg).into_dimensions().ok()
}

/// Writes a printable A4 index: a map of the mosaic split into areas, then
/// every photo with its thumbnail, date, file and where it is on the print.
/// Returns the number of photos listed.
pub fn write_index_pdf(cat: &Catalog, p: &Project, overview: Option<&RgbImage>, out: &Path, job: &Job) -> Result<usize> {
    let entries = build_index(cat, p, job)?;
    let areas = Areas::for_grid(p.grid.cols, p.grid.rows);
    let mut tmp = TempOutput::create(out)?;
    let mut pdf = PdfOut { w: BufWriter::with_capacity(1 << 20, tmp.file()), pos: 0, offsets: vec![0] };
    pdf.raw(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")?;
    let catalog = pdf.alloc();
    let pages_obj = pdf.alloc();
    let f1 = pdf.alloc();
    let f2 = pdf.alloc();
    pdf.obj(f1, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")?;
    pdf.obj(f2, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")?;
    let media = format!("[0 0 {:.4} {:.4}]", SHEET_W * PT_PER_MM, SHEET_H * PT_PER_MM);
    let fonts = format!("/Font << /F1 {f1} 0 R /F2 {f2} 0 R >>");
    let mut kids = Vec::new();
    let mut page = |pdf: &mut PdfOut<_>, c: Content, xobjects: &str| -> Result<()> {
        let content = pdf.alloc();
        pdf.stream(content, "", c.s.as_bytes())?;
        let pg = pdf.alloc();
        pdf.obj(pg, &format!("<< /Type /Page /Parent {pages_obj} 0 R /MediaBox {media} /Contents {content} 0 R /Resources << {fonts} /XObject << {xobjects} >> >> >>"))?;
        kids.push(pg);
        Ok(())
    };

    // --- Page 1: how to use it, and the map.
    {
        let mut c = Content::new(SHEET_H);
        let mut y = MARGIN + 8.0;
        c.text(MARGIN, y, 16.0, true, "Find your photo");
        y += 7.0;
        let (w, h) = (p.grid.width_mm, p.grid.height_mm);
        c.text(MARGIN, y, 8.5, false, &format!(
            "{} photos on {} tiles ({} columns × {} rows). Print size {} × {}.",
            entries.len(), p.tiles.iter().filter(|t| !t.is_empty()).count(), p.grid.cols, p.grid.rows, length(w), length(h)
        ));
        let steps = [
            "1. Find your photo in the list on the next pages. The list is in date order.".to_string(),
            "2. Find its area on the map below. Area rows have letters (A is at the top). Area columns have numbers (1 is at the left).".to_string(),
            format!("3. Each area has {0} × {0} tiles. Look in that part of the print, or measure from the top-left corner.", areas.tiles),
        ];
        y += 3.0;
        for s in steps {
            y += 4.5;
            c.text(MARGIN, y, 8.0, false, &s);
        }
        y += 6.0;
        let (avail_w, avail_h) = (SHEET_W - 2.0 * MARGIN - 8.0, SHEET_H - y - MARGIN - 8.0);
        let s = (avail_w / w).min(avail_h / h);
        let (mw, mh) = (w * s, h * s);
        let (mx, my) = (MARGIN + 8.0 + (avail_w - mw) / 2.0, y + 6.0);
        let mut xobj = String::new();
        if let Some(img) = overview {
            let jpg = crate::decode::encode_jpeg(img, 85)?;
            let im = pdf.alloc();
            pdf.stream(im, &format!("/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode", img.width(), img.height()), &jpg)?;
            xobj = format!("/Map {im} 0 R");
            c.image("Map", mx, my, mw, mh);
        }
        // Area grid with labels outside the map.
        let (tw, th) = (mw / p.grid.cols.max(1) as f64, mh / p.grid.rows.max(1) as f64);
        let (aw, ah) = (tw * areas.tiles as f64, th * areas.tiles as f64);
        c.s += "1 1 1 RG\n";
        for i in 1..areas.across {
            c.line(mx + i as f64 * aw, my, mx + i as f64 * aw, my + mh, 1.2);
        }
        for j in 1..areas.down {
            c.line(mx, my + j as f64 * ah, mx + mw, my + j as f64 * ah, 1.2);
        }
        c.s += "0.85 0 0 RG\n";
        for i in 1..areas.across {
            c.line(mx + i as f64 * aw, my, mx + i as f64 * aw, my + mh, 0.5);
        }
        for j in 1..areas.down {
            c.line(mx, my + j as f64 * ah, mx + mw, my + j as f64 * ah, 0.5);
        }
        c.s += "0 0 0 RG\n";
        c.rect(mx, my, mw, mh, 0.5);
        let fs = (aw.min(ah) * 1.4).clamp(5.0, 10.0);
        for i in 0..areas.across {
            let cx = mx + i as f64 * aw + aw.min(mw - i as f64 * aw) / 2.0;
            let l = (i + 1).to_string();
            c.text(cx - text_w(&l, fs) / 2.0, my - 1.8, fs, true, &l);
        }
        for j in 0..areas.down {
            let cy = my + j as f64 * ah + ah.min(mh - j as f64 * ah) / 2.0;
            let l = row_label(j);
            c.text(mx - 2.0 - text_w(&l, fs), cy + fs * 0.35 / PT_PER_MM, fs, true, &l);
        }
        page(&mut pdf, c, &xobj)?;
    }

    // --- The list: two columns of photos per page.
    let col_w = (SHEET_W - 2.0 * MARGIN - COL_GAP) / 2.0;
    let per_col = ((SHEET_H - 2.0 * MARGIN - 8.0) / ROW_H).floor().max(1.0) as usize;
    let per_page = per_col * 2;
    let pages = entries.len().div_ceil(per_page);
    for (pi, chunk) in entries.chunks(per_page).enumerate() {
        job.check()?;
        job.report("writing photo index", pi as u64, pages as u64);
        let mut c = Content::new(SHEET_H);
        let mut xobj = String::new();
        c.text(MARGIN, MARGIN + 3.0, 7.0, true, "Find your photo");
        let n = format!("Page {} of {}", pi + 2, pages + 1);
        c.text(SHEET_W - MARGIN - text_w(&n, 7.0), MARGIN + 3.0, 7.0, false, &n);
        c.s += "0.6 G\n";
        c.line(MARGIN, MARGIN + 4.5, SHEET_W - MARGIN, MARGIN + 4.5, 0.3);
        c.s += "0 G\n";
        for (k, e) in chunk.iter().enumerate() {
            let x = MARGIN + (k / per_col) as f64 * (col_w + COL_GAP);
            let y = MARGIN + 8.0 + (k % per_col) as f64 * ROW_H;
            // Thumbnail, as stored in the catalogue (already a JPEG).
            let thumb = cat.thumb(&unhex(&p.sources[e.source].hash)?)?;
            if let Some((bytes, (iw, ih))) = thumb.and_then(|b| jpeg_size(&b).map(|d| (b, d))) {
                let im = pdf.alloc();
                pdf.stream(im, &format!("/Type /XObject /Subtype /Image /Width {iw} /Height {ih} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode"), &bytes)?;
                let name = format!("T{k}");
                xobj += &format!("/{name} {im} 0 R ");
                let s = THUMB / iw.max(ih) as f64;
                let (w, h) = (iw as f64 * s, ih as f64 * s);
                c.image(&name, x + (THUMB - w) / 2.0, y + (THUMB - h) / 2.0, w, h);
            } else {
                c.s += "0.8 G\n";
                c.rect(x, y, THUMB, THUMB, 0.4);
                c.s += "0 G\n";
            }
            let tx = x + THUMB + 2.5;
            let tw = col_w - THUMB - 2.5;
            let mut ty = y + 2.6;
            c.text(tx, ty, 7.5, true, &fit(&e.name(), 7.5, tw, false));
            ty += 3.0;
            let date = e.taken().map(friendly_date).unwrap_or_else(|| "No date".into());
            let uses = if e.tiles.len() == 1 { "1 tile".to_string() } else { format!("{} tiles", e.tiles.len()) };
            c.text(tx, ty, 6.5, false, &format!("{date}  ·  {uses}"));
            if let Some(dir) = e.path().and_then(|p| p.parent()) {
                ty += 2.6;
                c.s += "0.45 g\n";
                c.text(tx, ty, 5.5, false, &fit(&dir.display().to_string(), 5.5, tw, true));
                c.s += "0 g\n";
            }
            for &t in e.tiles.iter().take(MAX_PLACES) {
                ty += 2.9;
                let pl = TilePlace::of(p, t);
                c.text(tx, ty, 6.5, true, &pl.area);
                c.text(tx + 6.5, ty, 6.0, false, &fit(&format!("row {}, col {}  ·  {}", pl.row, pl.col, pl.distance()), 6.0, tw - 6.5, false));
            }
            if e.tiles.len() > MAX_PLACES {
                ty += 2.9;
                // The other areas, in map order, each once.
                let cols = p.grid.cols.max(1) as usize;
                let set: std::collections::BTreeSet<(u32, u32)> = e.tiles[MAX_PLACES..].iter().map(|&t| ((t / cols) as u32 / areas.tiles, (t % cols) as u32 / areas.tiles)).collect();
                let more: Vec<String> = set.into_iter().map(|(r, c)| areas.label(r * areas.tiles, c * areas.tiles)).collect();
                c.text(tx, ty, 6.0, false, &fit(&format!("and {} more, in {}", e.tiles.len() - MAX_PLACES, more.join(", ")), 6.0, tw, false));
            }
        }
        page(&mut pdf, c, &xobj)?;
    }
    job.report("writing photo index", pages as u64, pages as u64);

    let kid_refs: Vec<String> = kids.iter().map(|k| format!("{k} 0 R")).collect();
    pdf.obj(pages_obj, &format!("<< /Type /Pages /Kids [{}] /Count {} >>", kid_refs.join(" "), kids.len()))?;
    pdf.obj(catalog, &format!("<< /Type /Catalog /Pages {pages_obj} 0 R >>"))?;
    let xref = pdf.pos;
    let mut x = format!("xref\n0 {}\n0000000000 65535 f \n", pdf.offsets.len());
    for o in &pdf.offsets[1..] {
        x += &format!("{o:010} 00000 n \n");
    }
    x += &format!("trailer\n<< /Size {} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n", pdf.offsets.len());
    pdf.raw(x.as_bytes())?;
    pdf.w.flush()?;
    drop(pdf);
    tmp.commit()?;
    Ok(entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn areas_cover_the_grid() {
        let a = Areas::for_grid(100, 80);
        assert_eq!((a.tiles, a.across, a.down), (10, 10, 8));
        assert_eq!(a.label(0, 0), "A1");
        assert_eq!(a.label(79, 99), "H10");
        let a = Areas::for_grid(7, 5);
        assert_eq!((a.tiles, a.across, a.down), (1, 7, 5));
        let a = Areas::for_grid(45, 30);
        assert_eq!((a.tiles, a.across, a.down), (5, 9, 6));
        assert_eq!(a.label(12, 23), "C5");
    }

    #[test]
    fn dates_are_easy_to_read() {
        assert_eq!(friendly_date("2023:07:14 15:02:11"), "14 Jul 2023, 15:02");
        assert_eq!(friendly_date("2023:07:14 00:00:00"), "14 Jul 2023");
        assert_eq!(friendly_date("2023-12-01"), "1 Dec 2023");
        assert_eq!(friendly_date("0000:00:00 00:00:00"), "0000:00:00 00:00:00");
        assert_eq!(friendly_date("summer"), "summer");
    }

    #[test]
    fn long_text_is_cut() {
        assert_eq!(fit("short", 7.0, 50.0, false), "short");
        let s = fit(&"x".repeat(200), 7.0, 30.0, false);
        assert!(s.ends_with("...") && text_w(&s, 7.0) <= 30.0 + 2.0, "{s}");
        let s = fit("/very/long/folder/path/to/the/photos/of/the/summer/holiday", 6.0, 20.0, true);
        assert!(s.starts_with("...") && s.ends_with("holiday"), "{s}");
    }

    #[test]
    fn distances_use_sensible_units() {
        assert_eq!(length(42.0), "42 mm");
        assert_eq!(length(415.0), "42 cm");
        assert_eq!(length_in(42.0, 1200.0), "4 cm");
        assert_eq!(length(12_345.0), "12.35 m");
    }
}
