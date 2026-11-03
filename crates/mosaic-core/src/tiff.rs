//! Minimal streaming writer for tiled BigTIFF (RGB8, Deflate + horizontal
//! predictor, or uncompressed).
//!
//! Why not the `tiff` crate: at the time of writing its encoder writes strips
//! only, and it is not needed to stream tiles band by band. This writer holds
//! one band of tiles plus the tile offset tables (16 bytes per tile; a 3 m ×
//! 3 m @ 300 DPI image with 256 px tiles has ~19,000 tiles = 300 KB).
//! Layout: header, tile data in row-major order, offset/count arrays, IFD.

use anyhow::{bail, Result};
use std::io::{Seek, SeekFrom, Write};

pub const TILE: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    None,
    Deflate,
}

pub struct TiledTiffWriter<W: Write + Seek> {
    w: W,
    width: u32,
    height: u32,
    dpi: f64,
    compression: TiffCompression,
    offsets: Vec<u64>,
    counts: Vec<u64>,
    pos: u64,
}

impl<W: Write + Seek> TiledTiffWriter<W> {
    pub fn new(mut w: W, width: u32, height: u32, dpi: f64, compression: TiffCompression) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("empty image");
        }
        // BigTIFF header: "II", 43, offset size 8, 0, first IFD offset (patched later).
        w.write_all(b"II")?;
        w.write_all(&43u16.to_le_bytes())?;
        w.write_all(&8u16.to_le_bytes())?;
        w.write_all(&0u16.to_le_bytes())?;
        w.write_all(&0u64.to_le_bytes())?;
        Ok(Self { w, width, height, dpi, compression, offsets: Vec::new(), counts: Vec::new(), pos: 16 })
    }

    pub fn tiles_across(&self) -> u32 {
        self.width.div_ceil(TILE)
    }
    pub fn tiles_down(&self) -> u32 {
        self.height.div_ceil(TILE)
    }

    /// Encode one tile (input: TILE×TILE RGB8, already padded). Thread-safe.
    pub fn encode_tile(compression: TiffCompression, tile: &[u8]) -> Result<Vec<u8>> {
        match compression {
            TiffCompression::None => Ok(tile.to_vec()),
            TiffCompression::Deflate => {
                // Horizontal differencing predictor (TIFF Predictor=2) per row.
                let row = TILE as usize * 3;
                let mut d = tile.to_vec();
                for r in d.chunks_exact_mut(row) {
                    for i in (3..row).rev() {
                        r[i] = r[i].wrapping_sub(r[i - 3]);
                    }
                }
                let mut enc = flate2::write::ZlibEncoder::new(Vec::with_capacity(tile.len() / 2), flate2::Compression::new(4));
                enc.write_all(&d)?;
                Ok(enc.finish()?)
            }
        }
    }

    /// Append encoded tiles in row-major order.
    pub fn write_encoded(&mut self, data: &[u8]) -> Result<()> {
        self.w.write_all(data)?;
        self.offsets.push(self.pos);
        self.counts.push(data.len() as u64);
        self.pos += data.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<W> {
        let expected = self.tiles_across() as usize * self.tiles_down() as usize;
        if self.offsets.len() != expected {
            bail!("TIFF incomplete: {} of {} tiles written", self.offsets.len(), expected);
        }
        if self.pos % 2 == 1 {
            self.w.write_all(&[0])?;
            self.pos += 1;
        }
        let off_pos = self.pos;
        for o in &self.offsets {
            self.w.write_all(&o.to_le_bytes())?;
        }
        let cnt_pos = off_pos + 8 * self.offsets.len() as u64;
        for c in &self.counts {
            self.w.write_all(&c.to_le_bytes())?;
        }
        let mut ifd_pos = cnt_pos + 8 * self.counts.len() as u64;
        // In BigTIFF, values of up to 8 bytes are stored inline in the entry:
        // RATIONAL (num, den) and the short Software string both fit.
        let num = (self.dpi * 1000.0).round() as u64;
        let rational = num | (1000u64 << 32);
        let soft = b"mosaic\0";
        let mut soft_inline = [0u8; 8];
        soft_inline[..soft.len()].copy_from_slice(soft);
        let soft_val = u64::from_le_bytes(soft_inline);
        if ifd_pos % 2 == 1 {
            self.w.write_all(&[0])?;
            ifd_pos += 1;
        }
        const SHORT: u16 = 3;
        const LONG: u16 = 4;
        const RATIONAL: u16 = 5;
        const ASCII: u16 = 2;
        const LONG8: u16 = 16;
        let n = self.offsets.len() as u64;
        let bps = (8u64) | (8u64 << 16) | (8u64 << 32);
        let comp = if self.compression == TiffCompression::Deflate { 8 } else { 1 };
        let mut entries: Vec<(u16, u16, u64, u64)> = vec![
            (256, LONG, 1, self.width as u64),
            (257, LONG, 1, self.height as u64),
            (258, SHORT, 3, bps),
            (259, SHORT, 1, comp),
            (262, SHORT, 1, 2),
            (277, SHORT, 1, 3),
            (282, RATIONAL, 1, rational),
            (283, RATIONAL, 1, rational),
            (284, SHORT, 1, 1),
            (296, SHORT, 1, 2),
            (305, ASCII, soft.len() as u64, soft_val),
            (322, LONG, 1, TILE as u64),
            (323, LONG, 1, TILE as u64),
            (324, LONG8, n, if n == 1 { self.offsets[0] } else { off_pos }),
            (325, LONG8, n, if n == 1 { self.counts[0] } else { cnt_pos }),
        ];
        if comp == 8 {
            entries.push((317, SHORT, 1, 2));
        }
        entries.sort_by_key(|e| e.0);
        // Arrays longer than 8 bytes (tile offsets/counts) use the offsets above.
        self.w.write_all(&(entries.len() as u64).to_le_bytes())?;
        for (tag, ty, count, value) in entries {
            self.w.write_all(&tag.to_le_bytes())?;
            self.w.write_all(&ty.to_le_bytes())?;
            self.w.write_all(&count.to_le_bytes())?;
            self.w.write_all(&value.to_le_bytes())?;
        }
        self.w.write_all(&0u64.to_le_bytes())?;
        self.w.flush()?;
        self.w.seek(SeekFrom::Start(8))?;
        self.w.write_all(&ifd_pos.to_le_bytes())?;
        self.w.flush()?;
        Ok(self.w)
    }
}

/// Bytes of a BigTIFF with the given tiles (upper bound when uncompressed).
pub fn uncompressed_size(width: u32, height: u32) -> u64 {
    let tiles = width.div_ceil(TILE) as u64 * height.div_ceil(TILE) as u64;
    tiles * (TILE as u64 * TILE as u64 * 3 + 16) + 512
}
