//! Tiled, print-ready PDF written directly from the renderer.
//!
//! Each print page gets its own JPEG image (DCTDecode) of exactly its artwork
//! window at the print DPI, streamed band by band into the file. Objects are
//! written as soon as they are complete and only their byte offsets are kept,
//! so memory stays bounded regardless of page count or artwork size.
//! (`pdf-writer` and similar crates assemble the whole document in memory,
//! which is why this small writer exists.)

use crate::atomic::{ensure_disk_space, TempOutput};
use crate::export::{cache_budget_for, write_jpeg_region, ExportReport, JPEG_MAX_SIDE};
use crate::job::Job;
use crate::layout::{compute_layout, Layout, PrintSettings};
use crate::project::Project;
use crate::render::{Renderer, TileSource};
use crate::units::{format_bytes, RasterSpec};
use anyhow::{bail, Result};
use image::RgbImage;
use serde::Serialize;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::time::Instant;

const PT_PER_MM: f64 = 72.0 / 25.4;

#[derive(Debug, Clone, Serialize)]
pub struct PdfPlan {
    pub layout: Layout,
    pub spec: RasterSpec,
    pub print: PrintSettings,
    pub max_page_px: (u64, u64),
    pub est_peak_memory: u64,
    pub est_output_bytes: u64,
    pub notes: Vec<String>,
}

impl PdfPlan {
    pub fn summary(&self) -> String {
        let l = &self.layout;
        let (sw, sh) = self.print.sheet_mm();
        let mut s = format!(
            "Artwork: {:.0} × {:.0} mm, printed at {} DPI ({} × {} px overall)\n\
             Paper: {} {} ({:.1} × {:.1} mm), margins {:?} mm, overlap {} mm, marks {}\n\
             Usable artwork per page: {:.1} × {:.1} mm\n\
             Print pages: {} ({} columns × {} rows) + 1 assembly overview page\n\
             Largest page image: {} × {} px\n\
             Estimated peak memory: {}, estimated PDF size: {}\n",
            l.art_w,
            l.art_h,
            self.spec.dpi,
            self.spec.width_px,
            self.spec.height_px,
            self.print.paper_name,
            if self.print.landscape { "landscape" } else { "portrait" },
            sw,
            sh,
            self.print.margins_mm,
            self.print.overlap_mm,
            if self.print.marks { "on" } else { "off" },
            l.usable_w,
            l.usable_h,
            l.pages.len(),
            l.cols,
            l.rows,
            self.max_page_px.0,
            self.max_page_px.1,
            format_bytes(self.est_peak_memory),
            format_bytes(self.est_output_bytes),
        );
        for n in &self.notes {
            s += &format!("Note: {n}\n");
        }
        s
    }
}

pub fn plan_pdf(project: &Project, print: &PrintSettings) -> Result<PdfPlan> {
    let g = &project.grid;
    let layout = compute_layout(g.width_mm, g.height_mm, print)?;
    let spec = RasterSpec::new(g.width_mm, g.height_mm, print.dpi)?;
    let mut max_page_px = (0, 0);
    for p in &layout.pages {
        let w = spec.x_px(p.x1) - spec.x_px(p.x0);
        let h = spec.y_px(p.y1) - spec.y_px(p.y0);
        max_page_px = (max_page_px.0.max(w), max_page_px.1.max(h));
    }
    if max_page_px.0 > JPEG_MAX_SIDE || max_page_px.1 > JPEG_MAX_SIDE {
        bail!("page images would be {}×{} px, above the {JPEG_MAX_SIDE} px JPEG limit; lower the DPI", max_page_px.0, max_page_px.1);
    }
    let cache = cache_budget_for(&spec, project);
    let est_peak_memory = cache + max_page_px.0 * 256 * 3 * 2 + rayon::current_num_threads() as u64 * (16 << 20) + (64 << 20);
    let page_px: u64 = layout.pages.iter().map(|p| (spec.x_px(p.x1) - spec.x_px(p.x0)) * (spec.y_px(p.y1) - spec.y_px(p.y0))).sum();
    let est_output_bytes = page_px * 3 / 8 + 2_000_000;
    let notes = vec![
        "Print at 100% / \"Actual size\". Disable \"Fit to page\", \"Shrink oversized pages\" and borderless enlargement.".into(),
        "Margins must be at least your printer's unprintable border, or content will be clipped.".into(),
        "Colour: sRGB images, no printer ICC profile or CMYK conversion.".into(),
    ];
    Ok(PdfPlan { layout, spec, print: print.clone(), max_page_px, est_peak_memory, est_output_bytes, notes })
}

struct PdfOut<W: Write + Seek> {
    w: W,
    pos: u64,
    offsets: Vec<u64>,
}

impl<W: Write + Seek> PdfOut<W> {
    fn alloc(&mut self) -> usize {
        self.offsets.push(0);
        self.offsets.len() - 1
    }
    fn raw(&mut self, b: &[u8]) -> Result<()> {
        self.w.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }
    fn obj(&mut self, n: usize, body: &str) -> Result<()> {
        self.offsets[n] = self.pos;
        self.raw(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes())
    }
    fn stream(&mut self, n: usize, dict: &str, data: &[u8]) -> Result<()> {
        self.offsets[n] = self.pos;
        self.raw(format!("{n} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes())?;
        self.raw(data)?;
        self.raw(b"\nendstream\nendobj\n")
    }
}

/// Counts bytes written through it (for streamed JPEG lengths).
struct Counting<'a, W: Write> {
    w: &'a mut W,
    n: u64,
}
impl<W: Write> Write for Counting<'_, W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let n = self.w.write(b)?;
        self.n += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// PDF text string (WinAnsi), escaping specials and mapping a few symbols.
fn pdf_text(s: &str) -> String {
    let mut o = String::from("(");
    for c in s.chars() {
        match c {
            '(' | ')' | '\\' => {
                o.push('\\');
                o.push(c);
            }
            '×' => o.push_str("\\327"),
            '–' | '—' => o.push('-'),
            c if c.is_ascii() && !c.is_control() => o.push(c),
            _ => o.push('?'),
        }
    }
    o.push(')');
    o
}

struct Content {
    s: String,
    sheet_h: f64,
}

impl Content {
    fn new(sheet_h: f64) -> Self {
        Self { s: String::new(), sheet_h }
    }
    fn x(&self, mm: f64) -> f64 {
        mm * PT_PER_MM
    }
    fn y(&self, mm: f64) -> f64 {
        (self.sheet_h - mm) * PT_PER_MM
    }
    /// Line from top-left-based mm coordinates.
    fn line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, width_pt: f64) {
        self.s += &format!("{width_pt:.3} w {:.3} {:.3} m {:.3} {:.3} l S\n", self.x(x0), self.y(y0), self.x(x1), self.y(y1));
    }
    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, width_pt: f64) {
        self.s += &format!("{width_pt:.3} w {:.3} {:.3} {:.3} {:.3} re S\n", self.x(x), self.y(y + h), w * PT_PER_MM, h * PT_PER_MM);
    }
    fn text(&mut self, x: f64, y_baseline: f64, size: f64, bold: bool, t: &str) {
        self.s += &format!("BT /{} {size:.2} Tf {:.3} {:.3} Td {} Tj ET\n", if bold { "F2" } else { "F1" }, self.x(x), self.y(y_baseline), pdf_text(t));
    }
    fn image(&mut self, name: &str, x: f64, y: f64, w: f64, h: f64) {
        self.s += &format!("q {:.4} 0 0 {:.4} {:.4} {:.4} cm /{name} Do Q\n", w * PT_PER_MM, h * PT_PER_MM, self.x(x), self.y(y + h));
    }
    /// Ruler of `len` mm with 1 mm / 5 mm / 10 mm ticks.
    fn ruler(&mut self, x: f64, y: f64, len: f64) {
        self.line(x, y, x + len, y, 0.4);
        let n = len.round() as i64;
        for i in 0..=n {
            let t = if i % 10 == 0 { 2.5 } else if i % 5 == 0 { 1.6 } else { 0.9 };
            self.line(x + i as f64, y, x + i as f64, y - t, 0.25);
        }
    }
}

pub fn export_pdf(project: &Project, source: &dyn TileSource, overview: Option<&RgbImage>, plan: &PdfPlan, out: &Path, job: &Job) -> Result<ExportReport> {
    let t0 = Instant::now();
    ensure_disk_space(out, plan.est_output_bytes)?;
    let renderer = Renderer::new(project, plan.spec, source, plan.est_peak_memory.min(cache_budget_for(&plan.spec, project)))?;
    let mut tmp = TempOutput::create(out)?;
    let print = &plan.print;
    let layout = &plan.layout;
    let (sw, sh) = print.sheet_mm();
    let (ax, ay) = print.art_origin_mm();
    let mut pdf = PdfOut { w: BufWriter::with_capacity(1 << 20, tmp.file()), pos: 0, offsets: vec![0] };
    pdf.raw(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")?;
    let catalog = pdf.alloc();
    let pages_obj = pdf.alloc();
    let f1 = pdf.alloc();
    let f2 = pdf.alloc();
    pdf.obj(f1, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")?;
    pdf.obj(f2, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")?;
    let media = format!("[0 0 {:.4} {:.4}]", sw * PT_PER_MM, sh * PT_PER_MM);
    let mut kids = Vec::new();

    // --- Overview / assembly map page.
    {
        let mut c = Content::new(sh);
        let m = print.margins_mm;
        let left = m[3] + 4.0;
        let mut y = m[0] + 8.0;
        c.text(left, y, 14.0, true, "Photo mosaic - assembly map");
        y += 6.0;
        c.text(left, y, 8.0, false, &format!(
            "Final size {:.0} × {:.0} mm. {} pages ({} columns × {} rows) on {} {}. Overlap {} mm. {} DPI.",
            layout.art_w, layout.art_h, layout.pages.len(), layout.cols, layout.rows, print.paper_name,
            if print.landscape { "landscape" } else { "portrait" }, print.overlap_mm, print.dpi
        ));
        y += 4.0;
        let lines = [
            "1. Print every page at 100% / \"Actual size\". Turn OFF \"Fit to page\", \"Shrink oversized pages\" and borderless printing.",
            "2. Check scale: the square below must measure exactly 50 × 50 mm and the ruler 100 mm. If not, fix the printer settings first.",
            "3. Pages are labelled by row letter and column number; A1 is the top-left page. Each page names its neighbours.",
            "4. Trim the LEFT edge of every page except column 1, and the TOP edge of every page except row A, along the crop marks.",
            "5. Lay each trimmed page over its left/upper neighbour so the cut edge sits on that page's alignment ticks (the shared overlap).",
            "6. Finally trim the outer edges of the edge pages along their crop marks.",
        ];
        for l in lines {
            y += 4.0;
            c.text(left, y, 7.0, false, l);
        }
        y += 6.0;
        // Calibration square and ruler.
        let cal_fits = sw - m[1] - left > 120.0;
        if cal_fits {
            c.rect(left, y, 50.0, 50.0, 0.5);
            c.text(left + 12.0, y + 26.0, 8.0, true, "50 × 50 mm");
            c.ruler(left + 60.0, y + 10.0, 100.0f64.min(sw - m[1] - left - 64.0).floor());
            c.text(left + 60.0, y + 16.0, 7.0, false, "0");
            c.text(left + 156.0, y + 16.0, 7.0, false, "100 mm");
            y += 56.0;
        } else {
            c.rect(left, y, 50.0, 50.0, 0.5);
            c.text(left + 12.0, y + 26.0, 8.0, true, "50 × 50 mm");
            y += 56.0;
        }
        // Map: preview with page grid.
        let avail_w = sw - left - m[1] - 4.0;
        let avail_h = sh - y - m[2] - 6.0;
        let s = (avail_w / layout.art_w).min(avail_h / layout.art_h);
        if s > 0.0 {
            let (mw, mh) = (layout.art_w * s, layout.art_h * s);
            let mut resources_img = String::new();
            if let Some(img) = overview {
                let jpg = crate::decode::encode_jpeg(img, 85)?;
                let im = pdf.alloc();
                pdf.stream(im, &format!("/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode", img.width(), img.height()), &jpg)?;
                resources_img = format!("/XObject << /Ov {im} 0 R >>");
                c.image("Ov", left, y, mw, mh);
            }
            c.s += "0.85 0 0 RG\n";
            for p in &layout.pages {
                let (px, py) = (left + p.x0 * s, y + p.y0 * s);
                c.rect(px, py, (p.x1 - p.x0) * s, (p.y1 - p.y0) * s, 0.5);
                c.s += "0 0 0 rg\n";
                c.text(px + 1.0, py + 3.5, (2.8 * s * layout.usable_w / 40.0).clamp(4.0, 9.0), true, &p.label);
            }
            c.s += "0 0 0 RG\n";
            let content = pdf.alloc();
            pdf.stream(content, "", c.s.as_bytes())?;
            let page = pdf.alloc();
            pdf.obj(page, &format!("<< /Type /Page /Parent {pages_obj} 0 R /MediaBox {media} /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> {resources_img} >> >>"))?;
            kids.push(page);
        } else {
            bail!("paper too small for the assembly map");
        }
    }

    // --- Print pages.
    let spec = &plan.spec;
    let mmx = spec.mm_per_px_x();
    let mmy = spec.mm_per_px_y();
    let total = layout.pages.len() as u64;
    for (i, p) in layout.pages.iter().enumerate() {
        job.check()?;
        job.report("writing PDF pages", i as u64, total);
        let (px0, px1, py0, py1) = (spec.x_px(p.x0), spec.x_px(p.x1), spec.y_px(p.y0), spec.y_px(p.y1));
        let (iw, ih) = (px1 - px0, py1 - py0);
        // Image placement in sheet mm (snapped to the global pixel grid: no drift).
        let ix = ax + (px0 as f64 * mmx - p.x0);
        let iy = ay + (py0 as f64 * mmy - p.y0);
        let (imw, imh) = (iw as f64 * mmx, ih as f64 * mmy);
        // Streamed image XObject with an indirect length.
        let im = pdf.alloc();
        let len_obj = pdf.alloc();
        pdf.offsets[im] = pdf.pos;
        pdf.raw(format!("{im} 0 obj\n<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len_obj} 0 R >>\nstream\n").as_bytes())?;
        let n = {
            let mut cw = Counting { w: &mut pdf.w, n: 0 };
            write_jpeg_region(&mut cw, &renderer, px0, py0, iw, ih, print.jpeg_quality, print.dpi, job, &format!("page {} of {}", i + 1, total))?;
            cw.n
        };
        pdf.pos += n;
        pdf.raw(b"\nendstream\nendobj\n")?;
        pdf.obj(len_obj, &n.to_string())?;

        let mut c = Content::new(sh);
        c.image("Im", ix, iy, imw, imh);
        let (aw, ah) = (p.x1 - p.x0, p.y1 - p.y0);
        if print.marks {
            let gap = 1.0;
            let len = (print.mark_space_mm - gap - 0.5).max(1.0);
            let (l, t, r, b) = (ax, ay, ax + aw, ay + ah);
            // Crop marks at the artwork window corners.
            for &x in &[l, r] {
                c.line(x, t - gap, x, t - gap - len, 0.3);
                c.line(x, b + gap, x, b + gap + len, 0.3);
            }
            for &y in &[t, b] {
                c.line(l - gap, y, l - gap - len, y, 0.3);
                c.line(r + gap, y, r + gap + len, y, 0.3);
            }
            // Alignment ticks where the next page's trimmed edge goes.
            c.s += "0.85 0 0 RG\n";
            if p.col + 1 < layout.cols {
                let x = ax + (p.x0 + layout.step_x - p.x0);
                c.line(x, t - gap, x, t - gap - len, 0.6);
                c.line(x, b + gap, x, b + gap + len, 0.6);
            }
            if p.row + 1 < layout.rows {
                let y = ay + layout.step_y;
                c.line(l - gap, y, l - gap - len, y, 0.6);
                c.line(r + gap, y, r + gap + len, y, 0.6);
            }
            c.s += "0 0 0 RG\n";
            // Labels in the mark space below the artwork.
            let left_n = if p.col > 0 { format!("{}{}", crate::layout::row_label(p.row), p.col) } else { "-".into() };
            let right_n = if p.col + 1 < layout.cols { format!("{}{}", crate::layout::row_label(p.row), p.col + 2) } else { "-".into() };
            let up_n = if p.row > 0 { format!("{}{}", crate::layout::row_label(p.row - 1), p.col + 1) } else { "-".into() };
            let down_n = if p.row + 1 < layout.rows { format!("{}{}", crate::layout::row_label(p.row + 1), p.col + 1) } else { "-".into() };
            let ty = b + print.mark_space_mm - 1.5;
            c.text(l, ty, 6.5, true, &format!("{}", p.label));
            c.text(l + 9.0, ty, 5.0, false, &format!(
                "row {} col {} | left {left_n} right {right_n} up {up_n} down {down_n} | x {:.1}-{:.1} y {:.1}-{:.1} mm | print at 100%, no fit-to-page",
                crate::layout::row_label(p.row), p.col + 1, p.x0, p.x1, p.y0, p.y1
            ));
            // 50 mm calibration ruler in the top mark space, right aligned.
            if aw > 80.0 && print.mark_space_mm >= 4.0 {
                let ry = t - print.mark_space_mm * 0.35;
                c.ruler(r - 50.0, ry, 50.0);
                c.text(r - 72.0, ry - 0.3, 5.0, false, "50 mm ruler");
            }
        }
        let content = pdf.alloc();
        pdf.stream(content, "", c.s.as_bytes())?;
        let page = pdf.alloc();
        pdf.obj(page, &format!("<< /Type /Page /Parent {pages_obj} 0 R /MediaBox {media} /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> /XObject << /Im {im} 0 R >> >> >>"))?;
        kids.push(page);
    }
    job.report("writing PDF pages", total, total);
    let kid_refs: Vec<String> = kids.iter().map(|k| format!("{k} 0 R")).collect();
    pdf.obj(pages_obj, &format!("<< /Type /Pages /Kids [{}] /Count {} >>", kid_refs.join(" "), kids.len()))?;
    pdf.obj(catalog, &format!("<< /Type /Catalog /Pages {pages_obj} 0 R /ViewerPreferences << /PrintScaling /None >> >>"))?;
    let xref = pdf.pos;
    let mut x = format!("xref\n0 {}\n0000000000 65535 f \n", pdf.offsets.len());
    for o in &pdf.offsets[1..] {
        x += &format!("{o:010} 00000 n \n");
    }
    x += &format!("trailer\n<< /Size {} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n", pdf.offsets.len());
    pdf.raw(x.as_bytes())?;
    pdf.w.flush()?;
    drop(pdf);
    job.check()?;
    let bytes = tmp.file().metadata()?.len();
    tmp.commit()?;
    let st = renderer.stats();
    Ok(ExportReport {
        path: out.display().to_string(),
        bytes,
        elapsed_ms: t0.elapsed().as_millis(),
        width_px: spec.width_px,
        height_px: spec.height_px,
        cache_hits: st.cache_hits,
        cache_misses: st.cache_misses,
    })
}
