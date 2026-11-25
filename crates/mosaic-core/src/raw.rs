//! Pure-Rust embedded JPEG preview extraction for camera RAW files.
//!
//! Modern camera RAW formats (Canon CR2/CR3, Nikon NEF/NRW, Sony ARW, Adobe DNG,
//! Fujifilm RAF, Olympus ORF, Panasonic RW2, Pentax PEF, Samsung SRW, Leica RWL)
//! embed a camera-rendered JPEG preview (often full sensor resolution). Reading
//! that preview avoids heavy demosaicing and needs zero external C libraries,
//! while still giving sharp, colour-graded tiles with full DCT scaling support.

use anyhow::{bail, Result};
use std::io::Cursor;

/// File extensions recognised as camera RAW files.
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "nrw", "arw", "dng", "orf", "raf", "rw2", "pef", "srw", "rwl",
];

/// Returns true if `bytes` begins with an unmistakable camera RAW container header
/// (Fujifilm RAF, Canon CR3, Canon CR2, Olympus ORF, Panasonic RW2).
/// Standard TIFF-header RAWs (NEF, ARW, DNG, PEF, SRW, RWL) share TIFF magic
/// `II*\0` / `MM\0*` and are handled via [`extract_tiff_preview`].
pub fn is_raw_container(bytes: &[u8]) -> bool {
    if bytes.starts_with(b"FUJIFILMCCD-RAW ") {
        return true;
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" && &bytes[8..12] == b"crx " {
        return true;
    }
    if bytes.len() >= 4 && matches!(&bytes[0..4], b"IIRO" | b"IIRS" | b"IIU\0") {
        return true;
    }
    if bytes.len() >= 10 && matches!(&bytes[0..4], b"II*\0" | b"MM\0*") && &bytes[8..10] == b"CR" {
        return true;
    }
    false
}

/// Extract the largest valid embedded JPEG preview slice from a camera RAW file.
pub fn extract_preview_jpeg(bytes: &[u8]) -> Result<&[u8]> {
    // 1. Fujifilm RAF: header stores big-endian JPEG offset at 0x54 and length at 0x58.
    if bytes.starts_with(b"FUJIFILMCCD-RAW ") {
        if let Some(j) = extract_raf(bytes) {
            return Ok(j);
        }
    }
    // 2. Canon CR3 (ISO-BMFF): scan boxes (including PRVW / mdat) for largest JPEG.
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" && &bytes[8..12] == b"crx " {
        if let Some(j) = scan_largest_jpeg(bytes) {
            return Ok(j);
        }
        bail!("CR3 file contains no readable JPEG preview");
    }
    // 3. TIFF / TIFF-derived RAW (CR2, NEF, ARW, DNG, ORF, RW2, PEF, SRW, RWL):
    //    walk IFDs + SubIFDs first, then fall back to bounded SOI..EOI scan.
    if let Some(j) = extract_tiff_preview(bytes) {
        return Ok(j);
    }
    if is_raw_container(bytes) {
        if let Some(j) = scan_largest_jpeg(bytes) {
            return Ok(j);
        }
    }
    bail!("RAW file contains no readable embedded JPEG preview")
}

/// Extract a valid embedded JPEG preview from a TIFF-structured RAW file by
/// walking its IFD chain and SubIFDs (tags `0x0201`/`0x0202`, `0x0111`/`0x0117`,
/// Panasonic `0x002E` JpgFromRaw, and `0x014A` SubIFDs).
pub fn extract_tiff_preview(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 8 {
        return None;
    }
    let le = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let magic = read_u16(bytes, 2, le)?;
    // Classic TIFF (42), Olympus ORF (0x4F52 = 'RO' / 0x5352 = 'RS'), Panasonic RW2 (0x0055).
    if !matches!(magic, 42 | 0x4F52 | 0x5352 | 0x0055) {
        return None;
    }
    let first_ifd = read_u32(bytes, 4, le)? as usize;
    let mut queue = vec![first_ifd];
    let mut visited = std::collections::HashSet::new();
    let mut candidates: Vec<&[u8]> = Vec::new();
    let mut is_raw_tiff = magic != 42 || (bytes.len() >= 10 && &bytes[8..10] == b"CR");

    while let Some(off) = queue.pop() {
        if off == 0 || off + 2 > bytes.len() || !visited.insert(off) || visited.len() > 64 {
            continue;
        }
        let count = read_u16(bytes, off, le)? as usize;
        let entries_start = off + 2;
        let entries_end = entries_start.checked_add(count.checked_mul(12)?)?;
        if entries_end + 4 > bytes.len() {
            continue;
        }
        let mut jpeg_off: Option<usize> = None;
        let mut jpeg_len: Option<usize> = None;
        let mut strip_off: Option<usize> = None;
        let mut strip_len: Option<usize> = None;
        let mut compression: Option<u32> = None;

        for i in 0..count {
            let e = entries_start + i * 12;
            let tag = read_u16(bytes, e, le)?;
            let ty = read_u16(bytes, e + 2, le)?;
            let cnt = read_u32(bytes, e + 4, le)? as usize;
            let val_u32 = read_ifd_scalar(bytes, e, ty, le);
            match tag {
                // Compression
                0x0103 => {
                    compression = val_u32;
                    // Lossless JPEG / Nikon / Sony / Kodak RAW compression markers.
                    if matches!(val_u32, Some(7 | 32767 | 34713 | 65000)) {
                        is_raw_tiff = true;
                    }
                }
                // DNGVersion tag
                0xC612 => is_raw_tiff = true,
                // StripOffsets / StripByteCounts (single-strip JPEG preview in CR2/DNG/NEF)
                0x0111 if cnt == 1 => strip_off = val_u32.map(|v| v as usize),
                0x0117 if cnt == 1 => strip_len = val_u32.map(|v| v as usize),
                // JPEGInterchangeFormat / JPEGInterchangeFormatLength
                0x0201 => jpeg_off = val_u32.map(|v| v as usize),
                0x0202 => jpeg_len = val_u32.map(|v| v as usize),
                // Panasonic JpgFromRaw (tag 0x002E, UNDEFINED byte array)
                0x002E if cnt >= 64 => {
                    is_raw_tiff = true;
                    if let Some(o) = read_u32(bytes, e + 8, le).map(|v| v as usize) {
                        if let Some(slice) = bytes.get(o..o.saturating_add(cnt)) {
                            if is_jpeg_start(slice) {
                                candidates.push(slice);
                            }
                        }
                    }
                }
                // SubIFDs (0x014A) & ExifIFD (0x8769)
                0x014A => {
                    is_raw_tiff = true;
                    for o in read_ifd_u32_list(bytes, e, ty, cnt.min(16), le) {
                        queue.push(o as usize);
                    }
                }
                _ => {}
            }
        }
        if let (Some(o), Some(l)) = (jpeg_off, jpeg_len) {
            if let Some(slice) = bytes.get(o..o.saturating_add(l)) {
                if is_jpeg_start(slice) {
                    candidates.push(slice);
                }
            }
        }
        if let (Some(o), Some(l)) = (strip_off, strip_len) {
            if matches!(compression, Some(6 | 7) | None) {
                if let Some(slice) = bytes.get(o..o.saturating_add(l)) {
                    if is_jpeg_start(slice) {
                        candidates.push(slice);
                    }
                }
            }
        }
        let next_ifd = read_u32(bytes, entries_end, le)? as usize;
        if next_ifd != 0 {
            queue.push(next_ifd);
        }
    }

    if !is_raw_tiff && candidates.is_empty() {
        return None;
    }
    pick_largest_jpeg(candidates)
}

fn extract_raf(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 0x60 {
        return None;
    }
    let off = u32::from_be_bytes(bytes[0x54..0x58].try_into().ok()?) as usize;
    let len = u32::from_be_bytes(bytes[0x58..0x5C].try_into().ok()?) as usize;
    let slice = bytes.get(off..off.checked_add(len)?)?;
    jpeg_dimensions(slice).map(|_| slice)
}

/// Scan `bytes` for embedded JPEG streams (`FF D8 FF` .. `FF D9`) and return the
/// largest decodable preview by pixel area. Bounded so scanning a 50 MB RAW is fast.
fn scan_largest_jpeg(bytes: &[u8]) -> Option<&[u8]> {
    let mut candidates = Vec::new();
    let mut i = 0usize;
    while i + 4 < bytes.len() && candidates.len() < 32 {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xD8 && bytes[i + 2] == 0xFF
            && matches!(bytes[i + 3], 0xE0..=0xEF | 0xDB | 0xC0 | 0xC2 | 0xFE)
        {
            if let Some(end) = find_jpeg_end(bytes, i) {
                candidates.push(&bytes[i..end]);
                i = end;
                continue;
            }
        }
        i += 1;
    }
    pick_largest_jpeg(candidates)
}

/// Walk JPEG marker segments from `start` (`FF D8`) until `FF D9` (EOI).
fn find_jpeg_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start + 4 > bytes.len() || bytes[start] != 0xFF || bytes[start + 1] != 0xD8 {
        return None;
    }
    let mut pos = start + 2;
    let mut in_entropy = false;
    while pos + 1 < bytes.len() {
        if in_entropy {
            if bytes[pos] == 0xFF {
                let next = bytes[pos + 1];
                if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                    pos += 2;
                    continue;
                }
                if next == 0xFF {
                    pos += 1;
                    continue;
                }
                if next == 0xD9 {
                    return Some(pos + 2);
                }
                // Another marker ended the scan (e.g. progressive multi-scan JPEG).
                in_entropy = false;
            } else {
                pos += 1;
                continue;
            }
        }
        if bytes[pos] != 0xFF {
            return None;
        }
        while pos < bytes.len() && bytes[pos] == 0xFF {
            pos += 1;
        }
        if pos >= bytes.len() {
            return None;
        }
        let marker = bytes[pos];
        pos += 1;
        match marker {
            0xD9 => return Some(pos),
            0x01 | 0xD0..=0xD8 => {}
            0xDA => {
                // Start of Scan: skip header then scan entropy-coded data.
                if pos + 2 > bytes.len() {
                    return None;
                }
                let len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                if len < 2 || pos + len > bytes.len() {
                    return None;
                }
                pos += len;
                in_entropy = true;
            }
            _ => {
                if pos + 2 > bytes.len() {
                    return None;
                }
                let len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                if len < 2 || pos + len > bytes.len() {
                    return None;
                }
                pos += len;
            }
        }
    }
    None
}

fn is_jpeg_start(b: &[u8]) -> bool {
    b.len() >= 4 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

fn jpeg_dimensions(slice: &[u8]) -> Option<(u32, u32)> {
    if !is_jpeg_start(slice) {
        return None;
    }
    let mut dec = jpeg_decoder::Decoder::new(Cursor::new(slice));
    dec.read_info().ok()?;
    let info = dec.info()?;
    if info.width > 0 && info.height > 0 {
        Some((info.width as u32, info.height as u32))
    } else {
        None
    }
}

fn pick_largest_jpeg<'a>(candidates: Vec<&'a [u8]>) -> Option<&'a [u8]> {
    let mut best: Option<(u64, &'a [u8])> = None;
    for slice in candidates {
        if let Some((w, h)) = jpeg_dimensions(slice) {
            let area = w as u64 * h as u64;
            if best.map(|(ba, _)| area > ba).unwrap_or(true) {
                best = Some((area, slice));
            }
        }
    }
    best.map(|(_, s)| s)
}

fn read_u16(b: &[u8], off: usize, le: bool) -> Option<u16> {
    let s: [u8; 2] = b.get(off..off + 2)?.try_into().ok()?;
    Some(if le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) })
}

fn read_u32(b: &[u8], off: usize, le: bool) -> Option<u32> {
    let s: [u8; 4] = b.get(off..off + 4)?.try_into().ok()?;
    Some(if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
}

fn read_ifd_scalar(b: &[u8], entry: usize, ty: u16, le: bool) -> Option<u32> {
    match ty {
        3 => read_u16(b, entry + 8, le).map(|v| v as u32), // SHORT
        4 | 13 => read_u32(b, entry + 8, le),              // LONG / IFD
        _ => None,
    }
}

fn read_ifd_u32_list(b: &[u8], entry: usize, ty: u16, count: usize, le: bool) -> Vec<u32> {
    if count == 0 || !matches!(ty, 4 | 13) {
        return Vec::new();
    }
    if count == 1 {
        return read_u32(b, entry + 8, le).into_iter().collect();
    }
    let Some(off) = read_u32(b, entry + 8, le).map(|v| v as usize) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        if let Some(v) = read_u32(b, off + i * 4, le) {
            out.push(v);
        }
    }
    out
}

/// Read EXIF orientation (1..=8) from a TIFF/RAW container, if present.
pub fn container_exif_orientation(bytes: &[u8]) -> Option<u8> {
    let ex = exif::Reader::new().read_from_container(&mut Cursor::new(bytes)).ok()?;
    ex.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .and_then(|v| u8::try_from(v).ok())
        .filter(|v| (1..=8).contains(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode;
    use image::RgbImage;

    fn make_jpeg(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let img = RgbImage::from_pixel(w, h, image::Rgb(rgb));
        decode::encode_jpeg(&img, 90).unwrap()
    }

    #[test]
    fn extracts_fujifilm_raf_embedded_jpeg() {
        let jpg = make_jpeg(160, 120, [210, 80, 40]);
        let mut raf = vec![0u8; 0x80];
        raf[..16].copy_from_slice(b"FUJIFILMCCD-RAW ");
        let off = raf.len() as u32;
        raf[0x54..0x58].copy_from_slice(&off.to_be_bytes());
        raf[0x58..0x5C].copy_from_slice(&(jpg.len() as u32).to_be_bytes());
        raf.extend_from_slice(&jpg);

        assert!(is_raw_container(&raf));
        let extracted = extract_preview_jpeg(&raf).unwrap();
        assert_eq!(extracted, jpg.as_slice());
        let decoded = decode::decode_bytes(&raf, None).unwrap();
        assert_eq!(decoded.info.full_width, 160);
        assert_eq!(decoded.info.full_height, 120);
    }

    #[test]
    fn extracts_canon_cr3_largest_embedded_jpeg() {
        let small = make_jpeg(40, 30, [20, 20, 20]);
        let large = make_jpeg(200, 150, [40, 180, 90]);
        let mut cr3 = Vec::new();
        // Minimal ISO-BMFF ftyp box with brand "crx "
        cr3.extend_from_slice(&24u32.to_be_bytes());
        cr3.extend_from_slice(b"ftypcrx ");
        cr3.extend_from_slice(&[0u8; 12]);
        // Embed both a tiny thumbnail and a larger preview JPEG.
        cr3.extend_from_slice(&small);
        cr3.extend_from_slice(&[0u8; 32]);
        cr3.extend_from_slice(&large);

        assert!(is_raw_container(&cr3));
        let extracted = extract_preview_jpeg(&cr3).unwrap();
        assert_eq!(extracted, large.as_slice());
        let decoded = decode::decode_bytes(&cr3, None).unwrap();
        assert_eq!((decoded.info.full_width, decoded.info.full_height), (200, 150));
    }

    #[test]
    fn extracts_tiff_raw_subifd_largest_jpeg_preview() {
        let thumb = make_jpeg(48, 36, [10, 10, 10]);
        let preview = make_jpeg(240, 180, [30, 120, 220]);
        // Build a minimal little-endian TIFF/CR2 with IFD0 pointing to SubIFD and a thumbnail.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II*\0");
        buf.extend_from_slice(&16u32.to_le_bytes()); // IFD0 at offset 16
        buf.extend_from_slice(b"CR\x02\x00"); // CR2 header marker
        buf.resize(16, 0);

        // IFD0 at 16: 3 entries (JPEGInterchangeFormat, JPEGInterchangeFormatLength, SubIFDs)
        let ifd0_size = 2 + 3 * 12 + 4;
        let subifd_off = 16 + ifd0_size;
        let subifd_size = 2 + 2 * 12 + 4;
        let thumb_off = subifd_off + subifd_size;
        let preview_off = thumb_off + thumb.len();

        let write_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, cnt: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&cnt.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&3u16.to_le_bytes());
        write_entry(&mut buf, 0x014A, 4, 1, subifd_off as u32);
        write_entry(&mut buf, 0x0201, 4, 1, thumb_off as u32);
        write_entry(&mut buf, 0x0202, 4, 1, thumb.len() as u32);
        buf.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

        // SubIFD at subifd_off: 2 entries pointing to full preview JPEG
        assert_eq!(buf.len(), subifd_off);
        buf.extend_from_slice(&2u16.to_le_bytes());
        write_entry(&mut buf, 0x0201, 4, 1, preview_off as u32);
        write_entry(&mut buf, 0x0202, 4, 1, preview.len() as u32);
        buf.extend_from_slice(&0u32.to_le_bytes());

        assert_eq!(buf.len(), thumb_off);
        buf.extend_from_slice(&thumb);
        assert_eq!(buf.len(), preview_off);
        buf.extend_from_slice(&preview);

        assert!(is_raw_container(&buf));
        let extracted = extract_preview_jpeg(&buf).unwrap();
        assert_eq!(extracted, preview.as_slice());
        let decoded = decode::decode_bytes(&buf, None).unwrap();
        assert_eq!((decoded.info.full_width, decoded.info.full_height), (240, 180));
    }
}
