//! Colour conversions. All perceptual matching happens in CIELAB (D65),
//! computed from sRGB. Averaging is done in linear light.

use std::sync::OnceLock;

pub type Lab = [f32; 3];

fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f64 / 255.0;
            *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } as f32;
        }
        t
    })
}

#[inline]
pub fn srgb8_to_linear(v: u8) -> f32 {
    srgb_to_linear_lut()[v as usize]
}

fn linear_to_srgb_lut() -> &'static [u8; 4097] {
    static LUT: OnceLock<[u8; 4097]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0u8; 4097];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f64 / 4096.0;
            let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
            *v = (s * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        t
    })
}

#[inline]
pub fn linear_to_srgb8(v: f32) -> u8 {
    let i = (v.clamp(0.0, 1.0) * 4096.0 + 0.5) as usize;
    linear_to_srgb_lut()[i.min(4096)]
}

const XN: f32 = 0.95047;
const YN: f32 = 1.0;
const ZN: f32 = 1.08883;

#[inline]
fn f_lab(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D * D * D { t.cbrt() } else { t / (3.0 * D * D) + 4.0 / 29.0 }
}
#[inline]
fn f_lab_inv(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D { t * t * t } else { 3.0 * D * D * (t - 4.0 / 29.0) }
}

pub fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> Lab {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let (fx, fy, fz) = (f_lab(x / XN), f_lab(y / YN), f_lab(z / ZN));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn lab_to_linear_rgb(lab: Lab) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    let (x, y, z) = (XN * f_lab_inv(fx), YN * f_lab_inv(fy), ZN * f_lab_inv(fz));
    [
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    ]
}

pub fn srgb8_to_lab(rgb: [u8; 3]) -> Lab {
    linear_rgb_to_lab(srgb8_to_linear(rgb[0]), srgb8_to_linear(rgb[1]), srgb8_to_linear(rgb[2]))
}

pub fn lab_to_srgb8(lab: Lab) -> [u8; 3] {
    let l = lab_to_linear_rgb(lab);
    [linear_to_srgb8(l[0]), linear_to_srgb8(l[1]), linear_to_srgb8(l[2])]
}

/// Squared Euclidean distance in Lab (CIE76 ΔE²).
#[inline]
pub fn de2(a: &Lab, b: &Lab) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Quantise Lab into 3 bytes (L*2.55, a+128, b+128) for compact storage.
#[inline]
pub fn quantize_lab(l: Lab) -> [u8; 3] {
    [
        (l[0] * 2.55).round().clamp(0.0, 255.0) as u8,
        (l[1] + 128.0).round().clamp(0.0, 255.0) as u8,
        (l[2] + 128.0).round().clamp(0.0, 255.0) as u8,
    ]
}
#[inline]
pub fn dequantize_lab(q: [u8; 3]) -> Lab {
    [q[0] as f32 / 2.55, q[1] as f32 - 128.0, q[2] as f32 - 128.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lab_roundtrip_and_reference_values() {
        let w = srgb8_to_lab([255, 255, 255]);
        assert!((w[0] - 100.0).abs() < 0.1 && w[1].abs() < 0.1 && w[2].abs() < 0.1);
        let r = srgb8_to_lab([255, 0, 0]);
        assert!((r[0] - 53.24).abs() < 0.1 && (r[1] - 80.09).abs() < 0.2);
        for c in [[12u8, 200, 99], [0, 0, 0], [250, 3, 128]] {
            assert_eq!(lab_to_srgb8(srgb8_to_lab(c)), c);
        }
    }
}
