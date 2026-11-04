//! Image decoding with consistent orientation, colour and alpha handling.
//!
//! Rules (identical for indexing, previews and exports):
//! * Format is detected from file content, not the extension (JPEG, PNG, WebP,
//!   TIFF). 16-bit images are reduced to 8 bits per channel.
//! * EXIF orientation is applied, so all later coordinates refer to the
//!   upright image.
//! * Embedded RGB ICC profiles are converted to sRGB (moxcms). Untagged images
//!   are assumed to be sRGB. Gray/CMYK profiles are not colour-managed.
//! * Transparency is composited over white (paper).
//! * JPEGs are decoded with DCT scaling (1/2, 1/4, 1/8) when a smaller size is
//!   enough, which keeps memory and time low for large camera files.

use anyhow::{anyhow, bail, Context, Result};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, RgbImage};
use std::io::Cursor;
use std::path::Path;

/// Refuse to allocate more than this for one decoded image.
pub const MAX_DECODE_BYTES: u64 = 768 * 1024 * 1024;
/// Refuse to read files larger than this.
pub const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Jpeg,
    Png,
    WebP,
    Tiff,
}

pub fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        Some(Format::Jpeg)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some(Format::Png)
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some(Format::WebP)
    } else if bytes.len() >= 4 && matches!(&bytes[0..4], b"II*\0" | b"MM\0*" | b"II+\0" | b"MM\0+") {
        // Classic TIFF and BigTIFF, little- and big-endian.
        Some(Format::Tiff)
    } else {
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecodeInfo {
    /// Upright full-resolution dimensions (after EXIF orientation).
    pub full_width: u32,
    pub full_height: u32,
    pub exif_orientation: u8,
    pub has_icc: bool,
    pub icc_converted: bool,
    pub had_alpha: bool,
}

#[derive(Debug)]
pub struct Decoded {
    pub image: RgbImage,
    pub info: DecodeInfo,
}

/// Decode a file. `min_short_side` lets the decoder return a smaller image
/// (never smaller than requested unless the source is smaller).
pub fn decode_file(path: &Path, min_short_side: Option<u32>) -> Result<Decoded> {
    let meta = std::fs::metadata(path).with_context(|| format!("cannot stat {}", path.display()))?;
    if meta.len() > MAX_FILE_BYTES {
        bail!("file too large ({} bytes)", meta.len());
    }
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    decode_bytes(&bytes, min_short_side)
}

pub fn decode_bytes(bytes: &[u8], min_short_side: Option<u32>) -> Result<Decoded> {
    match sniff(bytes) {
        Some(Format::Jpeg) => decode_jpeg(bytes, min_short_side),
        Some(f) => decode_generic(bytes, f),
        None => bail!("unsupported format"),
    }
}

fn decode_jpeg(bytes: &[u8], min_short_side: Option<u32>) -> Result<Decoded> {
    let mut dec = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    dec.set_max_decoding_buffer_size(MAX_DECODE_BYTES as usize);
    dec.read_info().map_err(|e| anyhow!("corrupt JPEG: {e}"))?;
    let info = dec.info().ok_or_else(|| anyhow!("corrupt JPEG: no header"))?;
    let (sw, sh) = (info.width as u32, info.height as u32);
    if sw == 0 || sh == 0 {
        bail!("corrupt JPEG: zero size");
    }
    let orientation = dec.exif_data().and_then(Orientation::from_exif_chunk).unwrap_or(Orientation::NoTransforms);
    let exif_orientation = orientation_to_exif(orientation);
    if let Some(min) = min_short_side {
        let short = sw.min(sh);
        if min < short {
            let s = min as f64 / short as f64;
            let rw = ((sw as f64 * s).ceil() as u32).clamp(1, u16::MAX as u32) as u16;
            let rh = ((sh as f64 * s).ceil() as u32).clamp(1, u16::MAX as u32) as u16;
            dec.scale(rw, rh).map_err(|e| anyhow!("JPEG scale: {e}"))?;
        }
    }
    let icc = dec.icc_profile();
    let pixels = dec.decode().map_err(|e| anyhow!("corrupt JPEG: {e}"))?;
    let info = dec.info().ok_or_else(|| anyhow!("corrupt JPEG"))?;
    let (w, h) = (info.width as u32, info.height as u32);
    let n = (w as usize) * (h as usize);
    let rgb: Vec<u8> = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        jpeg_decoder::PixelFormat::L16 => pixels.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        jpeg_decoder::PixelFormat::CMYK32 => pixels
            .chunks_exact(4)
            .flat_map(|c| {
                // jpeg-decoder returns Adobe-style (inverted) CMYK; naive conversion.
                let k = c[3] as u32;
                [(c[0] as u32 * k / 255) as u8, (c[1] as u32 * k / 255) as u8, (c[2] as u32 * k / 255) as u8]
            })
            .collect(),
    };
    if rgb.len() < n * 3 {
        bail!("corrupt JPEG: short pixel data");
    }
    let mut img = RgbImage::from_raw(w, h, rgb).ok_or_else(|| anyhow!("corrupt JPEG buffer"))?;
    let is_rgb = info.pixel_format == jpeg_decoder::PixelFormat::RGB24;
    let icc_converted = match (&icc, is_rgb) {
        (Some(p), true) => apply_icc(&mut img, p),
        _ => false,
    };
    let mut dynimg = DynamicImage::ImageRgb8(img);
    dynimg.apply_orientation(orientation);
    let (fw, fh) = oriented_dims(sw, sh, exif_orientation);
    Ok(Decoded {
        image: dynimg.into_rgb8(),
        info: DecodeInfo {
            full_width: fw,
            full_height: fh,
            exif_orientation,
            has_icc: icc.is_some(),
            icc_converted,
            had_alpha: false,
        },
    })
}

fn decode_generic(bytes: &[u8], f: Format) -> Result<Decoded> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes));
    reader.set_format(match f {
        Format::Png => image::ImageFormat::Png,
        Format::WebP => image::ImageFormat::WebP,
        Format::Jpeg => image::ImageFormat::Jpeg,
        Format::Tiff => image::ImageFormat::Tiff,
    });
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    let mut dec = reader.into_decoder().map_err(|e| anyhow!("corrupt {f:?}: {e}"))?;
    let orientation = dec.orientation().unwrap_or(Orientation::NoTransforms);
    let icc = dec.icc_profile().ok().flatten();
    let had_alpha = dec.color_type().has_alpha();
    let img = DynamicImage::from_decoder(dec).map_err(|e| anyhow!("corrupt {f:?}: {e}"))?;
    let (sw, sh) = (img.width(), img.height());
    let mut rgb = if had_alpha { composite_over_white(&img) } else { img.into_rgb8() };
    let icc_converted = match &icc {
        Some(p) => apply_icc(&mut rgb, p),
        None => false,
    };
    let mut dynimg = DynamicImage::ImageRgb8(rgb);
    dynimg.apply_orientation(orientation);
    let exif_orientation = orientation_to_exif(orientation);
    let (fw, fh) = oriented_dims(sw, sh, exif_orientation);
    Ok(Decoded {
        image: dynimg.into_rgb8(),
        info: DecodeInfo { full_width: fw, full_height: fh, exif_orientation, has_icc: icc.is_some(), icc_converted, had_alpha },
    })
}

fn composite_over_white(img: &DynamicImage) -> RgbImage {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = RgbImage::new(w, h);
    for (o, p) in out.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as u32;
        for c in 0..3 {
            o[c] = ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    out
}

/// Convert from an embedded RGB ICC profile to sRGB. Returns true if applied.
fn apply_icc(img: &mut RgbImage, icc: &[u8]) -> bool {
    let Ok(src) = moxcms::ColorProfile::new_from_slice(icc) else { return false };
    if src.color_space != moxcms::DataColorSpace::Rgb {
        return false;
    }
    let dst = moxcms::ColorProfile::new_srgb();
    let Ok(t) = src.create_transform_8bit(moxcms::Layout::Rgb, &dst, moxcms::Layout::Rgb, moxcms::TransformOptions::default()) else {
        return false;
    };
    let mut out = vec![0u8; img.as_raw().len()];
    if t.transform(img.as_raw(), &mut out).is_err() {
        return false;
    }
    img.copy_from_slice(&out);
    true
}

fn orientation_to_exif(o: Orientation) -> u8 {
    match o {
        Orientation::NoTransforms => 1,
        Orientation::FlipHorizontal => 2,
        Orientation::Rotate180 => 3,
        Orientation::FlipVertical => 4,
        Orientation::Rotate90FlipH => 5,
        Orientation::Rotate90 => 6,
        Orientation::Rotate270FlipH => 7,
        Orientation::Rotate270 => 8,
    }
}

fn oriented_dims(w: u32, h: u32, exif: u8) -> (u32, u32) {
    if (5..=8).contains(&exif) { (h, w) } else { (w, h) }
}

/// Resize an RGB image (optionally cropping a normalised rectangle first).
/// Same filter everywhere so previews and exports match.
pub fn resize_crop(src: &RgbImage, crop: [f32; 4], dst_w: u32, dst_h: u32) -> Result<RgbImage> {
    use fast_image_resize as fr;
    let (sw, sh) = src.dimensions();
    let cx = crop[0] as f64 * sw as f64;
    let cy = crop[1] as f64 * sh as f64;
    let cw = (crop[2] as f64 * sw as f64).max(1.0).min(sw as f64 - cx);
    let ch = (crop[3] as f64 * sh as f64).max(1.0).min(sh as f64 - cy);
    let src_view = fr::images::ImageRef::new(sw, sh, src.as_raw(), fr::PixelType::U8x3)?;
    let mut dst = fr::images::Image::new(dst_w.max(1), dst_h.max(1), fr::PixelType::U8x3);
    let opts = fr::ResizeOptions::new()
        .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom))
        .crop(cx, cy, cw, ch);
    fr::Resizer::new().resize(&src_view, &mut dst, &opts)?;
    RgbImage::from_raw(dst_w.max(1), dst_h.max(1), dst.into_vec()).ok_or_else(|| anyhow!("resize buffer"))
}

pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut out, quality);
    enc.encode(img.as_raw(), img.width() as u16, img.height() as u16, jpeg_encoder::ColorType::Rgb)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JPEG with an EXIF APP1 segment carrying the given orientation.
    pub fn jpeg_with_orientation(img: &RgbImage, orientation: u16) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = jpeg_encoder::Encoder::new(&mut out, 95);
        let mut exif = b"Exif\0\0".to_vec();
        exif.extend_from_slice(b"II*\0\x08\0\0\0"); // TIFF header, IFD at 8
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&0x0112u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&[0, 0]);
        exif.extend_from_slice(&0u32.to_le_bytes());
        enc.add_app_segment(1, exif).unwrap();
        enc.encode(img.as_raw(), img.width() as u16, img.height() as u16, jpeg_encoder::ColorType::Rgb).unwrap();
        out
    }

    fn marker_image() -> RgbImage {
        // 64x32 landscape, red block in the top-left corner, rest blue.
        RgbImage::from_fn(64, 32, |x, y| if x < 16 && y < 16 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) })
    }

    #[test]
    fn exif_orientation_is_applied() {
        let img = marker_image();
        // Orientation 6: stored image must be rotated 90° CW to display.
        let d = decode_bytes(&jpeg_with_orientation(&img, 6), None).unwrap();
        assert_eq!(d.image.dimensions(), (32, 64));
        assert_eq!((d.info.full_width, d.info.full_height), (32, 64));
        // Top-left red block moves to the top-right after a CW rotation.
        let p = d.image.get_pixel(28, 4);
        assert!(p[0] > 200 && p[2] < 60, "{p:?}");
        let p = d.image.get_pixel(4, 4);
        assert!(p[2] > 200, "{p:?}");
    }

    #[test]
    fn scaled_jpeg_keeps_orientation() {
        let img = RgbImage::from_fn(800, 400, |x, _| image::Rgb([(x / 4) as u8, 0, 0]));
        let d = decode_bytes(&jpeg_with_orientation(&img, 8), Some(50)).unwrap();
        // Rotated 270: 400 wide, 800 tall upright; decoded at 1/8 => 50x100.
        assert_eq!(d.image.dimensions(), (50, 100));
        assert_eq!((d.info.full_width, d.info.full_height), (400, 800));
    }

    #[test]
    fn png_alpha_composited_over_white() {
        let rgba = image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 0, 0]));
        let mut bytes = Vec::new();
        DynamicImage::ImageRgba8(rgba).write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
        let d = decode_bytes(&bytes, None).unwrap();
        assert!(d.info.had_alpha);
        assert_eq!(d.image.get_pixel(0, 0).0, [255, 255, 255]);
    }

    /// Write a TIFF with the `tiff` crate, optionally tagging Orientation.
    fn tiff_bytes<C: ::tiff::encoder::colortype::ColorType>(w: u32, h: u32, data: &[C::Inner], orientation: Option<u16>) -> Vec<u8>
    where
        [C::Inner]: ::tiff::encoder::TiffValue,
    {
        let mut out = Cursor::new(Vec::new());
        let mut enc = ::tiff::encoder::TiffEncoder::new(&mut out).unwrap();
        let mut img = enc.new_image::<C>(w, h).unwrap();
        if let Some(o) = orientation {
            img.encoder().write_tag(::tiff::tags::Tag::Orientation, o).unwrap();
        }
        img.write_data(data).unwrap();
        out.into_inner()
    }

    #[test]
    fn tiff_8bit_rgb() {
        let img = marker_image();
        let bytes = tiff_bytes::<::tiff::encoder::colortype::RGB8>(img.width(), img.height(), img.as_raw(), None);
        assert_eq!(sniff(&bytes), Some(Format::Tiff));
        let d = decode_bytes(&bytes, None).unwrap();
        assert_eq!(d.image.dimensions(), img.dimensions());
        assert_eq!(d.image.get_pixel(4, 4), img.get_pixel(4, 4));
    }

    #[test]
    fn tiff_16bit_reduced_to_8bit() {
        // 0xFFFF red, 0x8080 green, 0 blue.
        let data: Vec<u16> = (0..16).flat_map(|_| [0xFFFFu16, 0x8080, 0]).collect();
        let bytes = tiff_bytes::<::tiff::encoder::colortype::RGB16>(4, 4, &data, None);
        let d = decode_bytes(&bytes, None).unwrap();
        assert_eq!(d.image.get_pixel(1, 1).0, [255, 128, 0]);
    }

    #[test]
    fn tiff_16bit_alpha_composited_over_white() {
        let data: Vec<u16> = (0..16).flat_map(|_| [0u16, 0, 0, 0]).collect();
        let bytes = tiff_bytes::<::tiff::encoder::colortype::RGBA16>(4, 4, &data, None);
        let d = decode_bytes(&bytes, None).unwrap();
        assert!(d.info.had_alpha);
        assert_eq!(d.image.get_pixel(0, 0).0, [255, 255, 255]);
    }

    #[test]
    fn tiff_orientation_applied() {
        let img = marker_image();
        let bytes = tiff_bytes::<::tiff::encoder::colortype::RGB8>(img.width(), img.height(), img.as_raw(), Some(6));
        let d = decode_bytes(&bytes, None).unwrap();
        assert_eq!(d.info.exif_orientation, 6);
        assert_eq!(d.image.dimensions(), (img.height(), img.width()));
        assert_eq!((d.info.full_width, d.info.full_height), (img.height(), img.width()));
    }

    #[test]
    fn tiff_gray16() {
        let data = vec![0x4000u16; 16];
        let bytes = tiff_bytes::<::tiff::encoder::colortype::Gray16>(4, 4, &data, None);
        let d = decode_bytes(&bytes, None).unwrap();
        assert_eq!(d.image.get_pixel(0, 0).0, [64, 64, 64]);
    }

    #[test]
    fn tiff_exif_read_for_stats() {
        use ::tiff::tags::Tag;
        let img = marker_image();
        let mut out = Cursor::new(Vec::new());
        let mut enc = ::tiff::encoder::TiffEncoder::new(&mut out).unwrap();
        let mut im = enc.new_image::<::tiff::encoder::colortype::RGB8>(img.width(), img.height()).unwrap();
        im.encoder().write_tag(Tag::Make, "Nikon").unwrap();
        im.encoder().write_tag(Tag::Model, "Z 8").unwrap();
        im.encoder().write_tag(Tag::Artist, "Nhat").unwrap();
        im.write_data(img.as_raw()).unwrap();
        let ((artist, _), ex) = crate::indexer::read_exif(&out.into_inner());
        assert_eq!(artist.as_deref(), Some("Nhat"));
        assert_eq!(ex.camera.as_deref(), Some("Nikon Z 8"));
    }

    #[test]
    fn corrupt_and_unsupported() {
        assert!(decode_bytes(b"hello world", None).unwrap_err().to_string().contains("unsupported"));
        let mut j = jpeg_with_orientation(&marker_image(), 1);
        j.truncate(j.len() / 3);
        assert!(decode_bytes(&j, None).is_err());
    }
}
