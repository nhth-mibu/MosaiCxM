//! Native HEIC/HEIF image decoding across Linux, Windows, and macOS.
//!
//! - **macOS:** Uses Apple's built-in `ImageIO` and `CoreGraphics` frameworks
//!   (Apple Silicon hardware HEVC decoding, automatic EXIF orientation, Display P3
//!   to sRGB conversion, and scaled thumbnail decoding with zero bundled `.dylib`s).
//! - **Linux & Windows:** Loads `libheif` dynamically once (`libheif.so.1` on Linux,
//!   bundled `heif.dll` next to the executable or on `PATH` on Windows) and calls its
//!   stable C ABI directly with RAII guards, embedded-thumbnail fast paths for
//!   indexing, and ICC profile extraction for `moxcms` colour management.

use anyhow::Result;
use image::RgbImage;

/// File extensions recognised as HEIC/HEIF photos.
pub const HEIF_EXTENSIONS: &[&str] = &["heic", "heif", "hif"];

/// Decoded upright RGB8 HEIF/HEIC image plus metadata for [`crate::decode::Decoded`].
pub struct DecodedHeif {
    pub image: RgbImage,
    pub full_width: u32,
    pub full_height: u32,
    /// Raw embedded ICC profile (if extracted and not yet applied by the backend).
    pub icc: Option<Vec<u8>>,
    /// True if the platform decoder already colour-managed the pixels to sRGB (macOS ImageIO).
    pub already_srgb: bool,
    pub had_alpha: bool,
}

const HEIF_BRANDS: &[[u8; 4]] = &[
    *b"heic", *b"heix", *b"hevc", *b"hevx", *b"mif1", *b"msf1", *b"heim", *b"heis", *b"hevm", *b"hevs",
];

/// Returns true if `bytes` begins with an ISO-BMFF `ftyp` box declaring a HEIC/HEIF brand.
pub fn is_heif_container(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let box_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let end = if box_len >= 12 { box_len.min(bytes.len()).min(128) } else { bytes.len().min(32) };
    let major: [u8; 4] = [bytes[8], bytes[9], bytes[10], bytes[11]];
    // Exclude Canon CR3 (`crx `), which is handled as a camera RAW container.
    if &major == b"crx " {
        return false;
    }
    if HEIF_BRANDS.contains(&major) {
        return true;
    }
    let mut off = 16usize;
    while off + 4 <= end {
        let brand: [u8; 4] = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        if HEIF_BRANDS.contains(&brand) {
            return true;
        }
        off += 4;
    }
    false
}

/// Decode a HEIC/HEIF image from memory.
///
/// When `min_short_side` is `Some(req)`, backends may decode an embedded thumbnail
/// or hardware-scaled version whose short side is at least `req` pixels.
pub fn decode(bytes: &[u8], min_short_side: Option<u32>, max_decode_bytes: u64) -> Result<DecodedHeif> {
    #[cfg(target_os = "macos")]
    {
        macos::decode(bytes, min_short_side, max_decode_bytes)
    }
    #[cfg(not(target_os = "macos"))]
    {
        libheif_ffi::decode(bytes, min_short_side, max_decode_bytes)
    }
}

/// Returns true if the native HEIC/HEIF decoder backend is available at runtime.
pub fn is_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        libheif_ffi::api().is_ok()
    }
}

#[cfg(not(target_os = "macos"))]
mod libheif_ffi {
    use super::DecodedHeif;
    use anyhow::{anyhow, bail, Result};
    use image::RgbImage;
    use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
    use std::ptr;
    use std::sync::OnceLock;

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct HeifError {
        code: c_uint,
        subcode: c_uint,
        message: *const c_char,
    }

    impl HeifError {
        fn check(self, ctx: &str) -> Result<()> {
            if self.code == 0 {
                return Ok(());
            }
            let msg = if self.message.is_null() {
                "unknown libheif error".to_string()
            } else {
                unsafe { CStr::from_ptr(self.message) }.to_string_lossy().into_owned()
            };
            bail!("{ctx}: {msg} (code {}, subcode {})", self.code, self.subcode);
        }
    }

    const HEIF_COLORSPACE_RGB: c_uint = 1;
    const HEIF_CHROMA_INTERLEAVED_RGB: c_uint = 10;
    const HEIF_CHROMA_INTERLEAVED_RGBA: c_uint = 11;
    const HEIF_CHANNEL_INTERLEAVED: c_uint = 10;

    type FnInit = unsafe extern "C" fn(*mut c_void) -> HeifError;
    type FnContextAlloc = unsafe extern "C" fn() -> *mut c_void;
    type FnContextFree = unsafe extern "C" fn(*mut c_void);
    type FnReadFromMemory = unsafe extern "C" fn(*mut c_void, *const c_void, usize, *const c_void) -> HeifError;
    type FnGetPrimaryHandle = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> HeifError;
    type FnHandleRelease = unsafe extern "C" fn(*const c_void);
    type FnHandleInt = unsafe extern "C" fn(*const c_void) -> c_int;
    type FnThumbIds = unsafe extern "C" fn(*const c_void, *mut u32, c_int) -> c_int;
    type FnGetThumb = unsafe extern "C" fn(*const c_void, u32, *mut *mut c_void) -> HeifError;
    type FnRawProfileSize = unsafe extern "C" fn(*const c_void) -> usize;
    type FnRawProfile = unsafe extern "C" fn(*const c_void, *mut c_void) -> HeifError;
    type FnDecodeImage = unsafe extern "C" fn(*const c_void, *mut *mut c_void, c_uint, c_uint, *const c_void) -> HeifError;
    type FnImageDim = unsafe extern "C" fn(*const c_void, c_uint) -> c_int;
    type FnGetPlane = unsafe extern "C" fn(*const c_void, c_uint, *mut c_int) -> *const u8;
    type FnImageRelease = unsafe extern "C" fn(*const c_void);

    pub(super) struct LibHeifApi {
        context_alloc: FnContextAlloc,
        context_free: FnContextFree,
        read_from_memory: FnReadFromMemory,
        get_primary_handle: FnGetPrimaryHandle,
        handle_release: FnHandleRelease,
        handle_get_width: FnHandleInt,
        handle_get_height: FnHandleInt,
        handle_has_alpha: FnHandleInt,
        handle_num_thumbs: Option<FnHandleInt>,
        handle_thumb_ids: Option<FnThumbIds>,
        handle_get_thumb: Option<FnGetThumb>,
        handle_raw_profile_size: Option<FnRawProfileSize>,
        handle_raw_profile: Option<FnRawProfile>,
        decode_image: FnDecodeImage,
        image_get_width: FnImageDim,
        image_get_height: FnImageDim,
        image_get_plane: FnGetPlane,
        image_release: FnImageRelease,
    }

    // Function pointers loaded from a process-global shared library are safe to share across threads.
    unsafe impl Send for LibHeifApi {}
    unsafe impl Sync for LibHeifApi {}

    static API: OnceLock<std::result::Result<LibHeifApi, String>> = OnceLock::new();

    #[cfg(unix)]
    mod dl {
        use std::ffi::{c_char, c_int, c_void};

        const RTLD_NOW: c_int = 2;

        unsafe extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }

        pub fn open_libheif() -> Option<*mut c_void> {
            let candidates: &[&[u8]] = &[
                b"libheif.so.1\0",
                b"libheif.so\0",
                b"/usr/lib/x86_64-linux-gnu/libheif.so.1\0",
                b"/usr/lib/aarch64-linux-gnu/libheif.so.1\0",
                b"/usr/lib64/libheif.so.1\0",
            ];
            for name in candidates {
                let h = unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_NOW) };
                if !h.is_null() {
                    return Some(h);
                }
            }
            None
        }

        pub unsafe fn sym(handle: *mut c_void, name: &[u8]) -> *mut c_void {
            unsafe { dlsym(handle, name.as_ptr() as *const c_char) }
        }

        pub fn missing_hint() -> &'static str {
            "HEIC/HEIF decoding requires libheif (install `libheif1` on Ubuntu/Debian or `libheif` on Fedora/Arch)"
        }
    }

    #[cfg(windows)]
    mod dl {
        use std::ffi::{c_char, c_void};
        use std::os::windows::ffi::OsStrExt;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryW(lp_lib_file_name: *const u16) -> *mut c_void;
            fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
        }

        fn try_load(path: &std::ffi::OsStr) -> Option<*mut c_void> {
            let wide: Vec<u16> = path.encode_wide().chain(Some(0)).collect();
            let h = unsafe { LoadLibraryW(wide.as_ptr()) };
            (!h.is_null()).then_some(h)
        }

        pub fn open_libheif() -> Option<*mut c_void> {
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    for name in ["heif.dll", "libheif.dll"] {
                        let p = dir.join(name);
                        if p.exists() {
                            if let Some(h) = try_load(p.as_os_str()) {
                                return Some(h);
                            }
                        }
                    }
                }
            }
            for name in ["heif.dll", "libheif.dll"] {
                if let Some(h) = try_load(std::ffi::OsStr::new(name)) {
                    return Some(h);
                }
            }
            None
        }

        pub unsafe fn sym(handle: *mut c_void, name: &[u8]) -> *mut c_void {
            unsafe { GetProcAddress(handle, name.as_ptr() as *const c_char) }
        }

        pub fn missing_hint() -> &'static str {
            "HEIC/HEIF decoding requires heif.dll (re-extract the release zip so heif.dll and libde265.dll sit next to mosaic-gui.exe)"
        }
    }

    unsafe fn load_req<T: Copy>(handle: *mut c_void, name: &[u8]) -> std::result::Result<T, String> {
        let ptr = unsafe { dl::sym(handle, name) };
        if ptr.is_null() {
            let s = String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]);
            return Err(format!("libheif is missing required symbol `{s}`"));
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&ptr) })
    }

    unsafe fn load_opt<T: Copy>(handle: *mut c_void, name: &[u8]) -> Option<T> {
        let ptr = unsafe { dl::sym(handle, name) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&ptr) })
        }
    }

    pub(super) fn api() -> Result<&'static LibHeifApi> {
        let res = API.get_or_init(|| {
            let Some(handle) = dl::open_libheif() else {
                return Err(dl::missing_hint().to_string());
            };
            unsafe {
                // Initialize libheif plugin registry on libheif >= 1.13 (optional on older builds).
                if let Some(init) = load_opt::<FnInit>(handle, b"heif_init\0") {
                    let _ = init(ptr::null_mut());
                }
                Ok(LibHeifApi {
                    context_alloc: load_req(handle, b"heif_context_alloc\0")?,
                    context_free: load_req(handle, b"heif_context_free\0")?,
                    read_from_memory: load_req(handle, b"heif_context_read_from_memory_without_copy\0")?,
                    get_primary_handle: load_req(handle, b"heif_context_get_primary_image_handle\0")?,
                    handle_release: load_req(handle, b"heif_image_handle_release\0")?,
                    handle_get_width: load_req(handle, b"heif_image_handle_get_width\0")?,
                    handle_get_height: load_req(handle, b"heif_image_handle_get_height\0")?,
                    handle_has_alpha: load_req(handle, b"heif_image_handle_has_alpha_channel\0")?,
                    handle_num_thumbs: load_opt(handle, b"heif_image_handle_get_number_of_thumbnails\0"),
                    handle_thumb_ids: load_opt(handle, b"heif_image_handle_get_list_of_thumbnail_IDs\0"),
                    handle_get_thumb: load_opt(handle, b"heif_image_handle_get_thumbnail\0"),
                    handle_raw_profile_size: load_opt(handle, b"heif_image_handle_get_raw_color_profile_size\0"),
                    handle_raw_profile: load_opt(handle, b"heif_image_handle_get_raw_color_profile\0"),
                    decode_image: load_req(handle, b"heif_decode_image\0")?,
                    image_get_width: load_req(handle, b"heif_image_get_width\0")?,
                    image_get_height: load_req(handle, b"heif_image_get_height\0")?,
                    image_get_plane: load_req(handle, b"heif_image_get_plane_readonly\0")?,
                    image_release: load_req(handle, b"heif_image_release\0")?,
                })
            }
        });
        res.as_ref().map_err(|e| anyhow!("{e}"))
    }

    struct CtxGuard<'a>(&'a LibHeifApi, *mut c_void);
    impl Drop for CtxGuard<'_> {
        fn drop(&mut self) {
            if !self.1.is_null() {
                unsafe { (self.0.context_free)(self.1) };
            }
        }
    }

    struct HandleGuard<'a>(&'a LibHeifApi, *mut c_void);
    impl Drop for HandleGuard<'_> {
        fn drop(&mut self) {
            if !self.1.is_null() {
                unsafe { (self.0.handle_release)(self.1) };
            }
        }
    }

    struct ImgGuard<'a>(&'a LibHeifApi, *mut c_void);
    impl Drop for ImgGuard<'_> {
        fn drop(&mut self) {
            if !self.1.is_null() {
                unsafe { (self.0.image_release)(self.1) };
            }
        }
    }

    pub(super) fn decode(bytes: &[u8], min_short_side: Option<u32>, max_decode_bytes: u64) -> Result<DecodedHeif> {
        let a = api()?;
        let ctx_ptr = unsafe { (a.context_alloc)() };
        if ctx_ptr.is_null() {
            bail!("libheif failed to allocate context");
        }
        let ctx = CtxGuard(a, ctx_ptr);
        unsafe { (a.read_from_memory)(ctx.1, bytes.as_ptr() as *const c_void, bytes.len(), ptr::null()) }
            .check("corrupt HEIC/HEIF file")?;

        let mut handle_ptr: *mut c_void = ptr::null_mut();
        unsafe { (a.get_primary_handle)(ctx.1, &mut handle_ptr) }.check("cannot read primary HEIC/HEIF image")?;
        if handle_ptr.is_null() {
            bail!("HEIC/HEIF file has no primary image");
        }
        let primary = HandleGuard(a, handle_ptr);

        let full_w = unsafe { (a.handle_get_width)(primary.1) };
        let full_h = unsafe { (a.handle_get_height)(primary.1) };
        if full_w <= 0 || full_h <= 0 {
            bail!("corrupt HEIC/HEIF: invalid dimensions ({full_w}x{full_h})");
        }
        let (full_w, full_h) = (full_w as u32, full_h as u32);

        // Extract embedded ICC profile (e.g. Apple Display P3) from the primary handle.
        let icc = match (a.handle_raw_profile_size, a.handle_raw_profile) {
            (Some(get_sz), Some(get_data)) => {
                let sz = unsafe { get_sz(primary.1) };
                if sz > 0 && sz <= 16 * 1024 * 1024 {
                    let mut buf = vec![0u8; sz];
                    let err = unsafe { get_data(primary.1, buf.as_mut_ptr() as *mut c_void) };
                    (err.code == 0).then_some(buf)
                } else {
                    None
                }
            }
            _ => None,
        };

        // Indexing fast-path: if a smaller image is enough and the HEIC embeds a thumbnail
        // that is large enough with matching aspect ratio, decode that instead of 12–48 MP.
        let mut thumb_guard: Option<HandleGuard<'_>> = None;
        if let (Some(req), Some(num_fn), Some(ids_fn), Some(get_fn)) =
            (min_short_side, a.handle_num_thumbs, a.handle_thumb_ids, a.handle_get_thumb)
        {
            if full_w.min(full_h) > req.saturating_mul(2) {
                let count = unsafe { num_fn(primary.1) };
                if count > 0 {
                    let n = (count as usize).min(16);
                    let mut ids = vec![0u32; n];
                    let got = unsafe { ids_fn(primary.1, ids.as_mut_ptr(), n as c_int) }.max(0) as usize;
                    let mut best: Option<(u32, HandleGuard<'_>)> = None;
                    let main_aspect = full_w as f32 / full_h as f32;
                    for &id in &ids[..got.min(n)] {
                        let mut th: *mut c_void = ptr::null_mut();
                        if unsafe { get_fn(primary.1, id, &mut th) }.code == 0 && !th.is_null() {
                            let guard = HandleGuard(a, th);
                            let tw = unsafe { (a.handle_get_width)(guard.1) };
                            let th_h = unsafe { (a.handle_get_height)(guard.1) };
                            if tw > 0 && th_h > 0 {
                                let (tw, th_h) = (tw as u32, th_h as u32);
                                let short = tw.min(th_h);
                                let aspect = tw as f32 / th_h as f32;
                                let aspect_ok = (aspect / main_aspect - 1.0).abs() < 0.08;
                                if short >= req && aspect_ok && best.as_ref().is_none_or(|(bs, _)| short < *bs) {
                                    best = Some((short, guard));
                                }
                            }
                        }
                    }
                    thumb_guard = best.map(|(_, g)| g);
                }
            }
        }

        let decode_handle = thumb_guard.as_ref().map_or(primary.1, |g| g.1);
        let dec_w = unsafe { (a.handle_get_width)(decode_handle) }.max(0) as u64;
        let dec_h = unsafe { (a.handle_get_height)(decode_handle) }.max(0) as u64;
        let had_alpha = unsafe { (a.handle_has_alpha)(decode_handle) } != 0;
        let channels = if had_alpha { 4u64 } else { 3u64 };
        let est_bytes = dec_w.saturating_mul(dec_h).saturating_mul(channels);
        if est_bytes == 0 || est_bytes > max_decode_bytes {
            bail!("HEIC/HEIF image too large ({dec_w}x{dec_h})");
        }

        let chroma = if had_alpha { HEIF_CHROMA_INTERLEAVED_RGBA } else { HEIF_CHROMA_INTERLEAVED_RGB };
        let mut img_ptr: *mut c_void = ptr::null_mut();
        unsafe { (a.decode_image)(decode_handle, &mut img_ptr, HEIF_COLORSPACE_RGB, chroma, ptr::null()) }
            .check("failed to decode HEIC/HEIF pixels")?;
        if img_ptr.is_null() {
            bail!("libheif returned a null image");
        }
        let img = ImgGuard(a, img_ptr);

        let w = unsafe { (a.image_get_width)(img.1, HEIF_CHANNEL_INTERLEAVED) };
        let h = unsafe { (a.image_get_height)(img.1, HEIF_CHANNEL_INTERLEAVED) };
        if w <= 0 || h <= 0 {
            bail!("corrupt HEIC/HEIF decoded plane ({w}x{h})");
        }
        let (w, h) = (w as usize, h as usize);
        let mut stride: c_int = 0;
        let plane = unsafe { (a.image_get_plane)(img.1, HEIF_CHANNEL_INTERLEAVED, &mut stride) };
        let bpp = if had_alpha { 4usize } else { 3usize };
        if plane.is_null() || stride < (w * bpp) as c_int {
            bail!("corrupt HEIC/HEIF plane stride");
        }
        let stride = stride as usize;
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            let src_row = unsafe { std::slice::from_raw_parts(plane.add(y * stride), w * bpp) };
            let dst_row = &mut rgb[y * w * 3..(y + 1) * w * 3];
            if had_alpha {
                for x in 0..w {
                    let s = &src_row[x * 4..x * 4 + 4];
                    let alpha = s[3] as u32;
                    dst_row[x * 3] = ((s[0] as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
                    dst_row[x * 3 + 1] = ((s[1] as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
                    dst_row[x * 3 + 2] = ((s[2] as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
                }
            } else {
                dst_row.copy_from_slice(src_row);
            }
        }
        let image = RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| anyhow!("HEIC/HEIF RGB buffer"))?;
        Ok(DecodedHeif {
            image,
            full_width: full_w,
            full_height: full_h,
            icc,
            already_srgb: false,
            had_alpha,
        })
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::DecodedHeif;
    use anyhow::{anyhow, bail, Result};
    use image::RgbImage;
    use std::ffi::c_void;
    use std::ptr;

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CGRect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    }

    const K_CF_NUMBER_SINT32_TYPE: isize = 3;
    const K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST: u32 = 1;
    const K_CG_BITMAP_BYTE_ORDER_32_BIG: u32 = 4 << 12;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFBooleanTrue: *const c_void;
        static kCFBooleanFalse: *const c_void;
        static kCFTypeDictionaryKeyCallBacks: c_void;
        static kCFTypeDictionaryValueCallBacks: c_void;
        fn CFDataCreate(allocator: *const c_void, bytes: *const u8, length: isize) -> *const c_void;
        fn CFNumberCreate(allocator: *const c_void, the_type: isize, value_ptr: *const c_void) -> *const c_void;
        fn CFNumberGetValue(number: *const c_void, the_type: isize, value_ptr: *mut c_void) -> u8;
        fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            num_values: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *const c_void;
        fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    #[link(name = "ImageIO", kind = "framework")]
    unsafe extern "C" {
        static kCGImagePropertyPixelWidth: *const c_void;
        static kCGImagePropertyPixelHeight: *const c_void;
        static kCGImagePropertyOrientation: *const c_void;
        static kCGImagePropertyHasAlpha: *const c_void;
        static kCGImageSourceCreateThumbnailFromImageAlways: *const c_void;
        static kCGImageSourceCreateThumbnailWithTransform: *const c_void;
        static kCGImageSourceThumbnailMaxPixelSize: *const c_void;
        static kCGImageSourceShouldCache: *const c_void;
        fn CGImageSourceCreateWithData(data: *const c_void, options: *const c_void) -> *const c_void;
        fn CGImageSourceCopyPropertiesAtIndex(isrc: *const c_void, index: usize, options: *const c_void) -> *const c_void;
        fn CGImageSourceCreateThumbnailAtIndex(isrc: *const c_void, index: usize, options: *const c_void) -> *const c_void;
        fn CGImageSourceCreateImageAtIndex(isrc: *const c_void, index: usize, options: *const c_void) -> *const c_void;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        static kCGColorSpaceSRGB: *const c_void;
        fn CGImageGetWidth(image: *const c_void) -> usize;
        fn CGImageGetHeight(image: *const c_void) -> usize;
        fn CGImageRelease(image: *const c_void);
        fn CGColorSpaceCreateWithName(name: *const c_void) -> *const c_void;
        fn CGColorSpaceRelease(space: *const c_void);
        fn CGBitmapContextCreate(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: *const c_void,
            bitmap_info: u32,
        ) -> *const c_void;
        fn CGContextSetRGBFillColor(c: *const c_void, r: f64, g: f64, b: f64, a: f64);
        fn CGContextFillRect(c: *const c_void, rect: CGRect);
        fn CGContextDrawImage(c: *const c_void, rect: CGRect, image: *const c_void);
        fn CGContextRelease(c: *const c_void);
    }

    struct Cf(*const c_void);
    impl Drop for Cf {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }

    struct CgImage(*const c_void);
    impl Drop for CgImage {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CGImageRelease(self.0) };
            }
        }
    }

    struct CgColorSpace(*const c_void);
    impl Drop for CgColorSpace {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CGColorSpaceRelease(self.0) };
            }
        }
    }

    struct CgContext(*const c_void);
    impl Drop for CgContext {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CGContextRelease(self.0) };
            }
        }
    }

    fn dict_i32(dict: *const c_void, key: *const c_void) -> Option<i32> {
        let v = unsafe { CFDictionaryGetValue(dict, key) };
        if v.is_null() {
            return None;
        }
        let mut out = 0i32;
        let ok = unsafe { CFNumberGetValue(v, K_CF_NUMBER_SINT32_TYPE, &mut out as *mut i32 as *mut c_void) };
        (ok != 0).then_some(out)
    }

    pub(super) fn decode(bytes: &[u8], min_short_side: Option<u32>, max_decode_bytes: u64) -> Result<DecodedHeif> {
        let cf_data = Cf(unsafe { CFDataCreate(ptr::null(), bytes.as_ptr(), bytes.len() as isize) });
        if cf_data.0.is_null() {
            bail!("ImageIO failed to wrap HEIC bytes");
        }
        let src = Cf(unsafe { CGImageSourceCreateWithData(cf_data.0, ptr::null()) });
        if src.0.is_null() {
            bail!("corrupt HEIC/HEIF file");
        }

        let props = Cf(unsafe { CGImageSourceCopyPropertiesAtIndex(src.0, 0, ptr::null()) });
        let (mut full_w, mut full_h, had_alpha) = if !props.0.is_null() {
            let raw_w = dict_i32(props.0, unsafe { kCGImagePropertyPixelWidth }).unwrap_or(0).max(0) as u32;
            let raw_h = dict_i32(props.0, unsafe { kCGImagePropertyPixelHeight }).unwrap_or(0).max(0) as u32;
            let orient = dict_i32(props.0, unsafe { kCGImagePropertyOrientation }).unwrap_or(1);
            let (fw, fh) = if (5..=8).contains(&orient) { (raw_h, raw_w) } else { (raw_w, raw_h) };
            let alpha_ptr = unsafe { CFDictionaryGetValue(props.0, kCGImagePropertyHasAlpha) };
            let alpha = !alpha_ptr.is_null() && alpha_ptr == unsafe { kCFBooleanTrue };
            (fw, fh, alpha)
        } else {
            (0, 0, false)
        };

        let target_max_px: Option<i32> = if full_w > 0 && full_h > 0 {
            match min_short_side {
                Some(req) if full_w.min(full_h) > req => {
                    let long = ((full_w.max(full_h) as u64 * req as u64 + full_w.min(full_h) as u64 - 1)
                        / full_w.min(full_h) as u64)
                        .max(req as u64)
                        .min(full_w.max(full_h) as u64);
                    Some(long.min(i32::MAX as u64) as i32)
                }
                _ => Some(full_w.max(full_h).min(i32::MAX as u32) as i32),
            }
        } else {
            None
        };

        let cg_img = if let Some(max_px) = target_max_px {
            let num = Cf(unsafe { CFNumberCreate(ptr::null(), K_CF_NUMBER_SINT32_TYPE, &max_px as *const i32 as *const c_void) });
            let keys = unsafe {
                [
                    kCGImageSourceCreateThumbnailFromImageAlways,
                    kCGImageSourceCreateThumbnailWithTransform,
                    kCGImageSourceThumbnailMaxPixelSize,
                    kCGImageSourceShouldCache,
                ]
            };
            let vals = unsafe { [kCFBooleanTrue, kCFBooleanTrue, num.0, kCFBooleanFalse] };
            let opts = Cf(unsafe {
                CFDictionaryCreate(
                    ptr::null(),
                    keys.as_ptr(),
                    vals.as_ptr(),
                    keys.len() as isize,
                    &raw const kCFTypeDictionaryKeyCallBacks,
                    &raw const kCFTypeDictionaryValueCallBacks,
                )
            });
            let p = unsafe { CGImageSourceCreateThumbnailAtIndex(src.0, 0, opts.0) };
            if !p.is_null() {
                CgImage(p)
            } else {
                CgImage(unsafe { CGImageSourceCreateImageAtIndex(src.0, 0, ptr::null()) })
            }
        } else {
            CgImage(unsafe { CGImageSourceCreateImageAtIndex(src.0, 0, ptr::null()) })
        };
        if cg_img.0.is_null() {
            bail!("ImageIO failed to decode HEIC/HEIF image");
        }

        let w = unsafe { CGImageGetWidth(cg_img.0) };
        let h = unsafe { CGImageGetHeight(cg_img.0) };
        if w == 0 || h == 0 {
            bail!("corrupt HEIC/HEIF: zero size");
        }
        if full_w == 0 || full_h == 0 {
            full_w = w as u32;
            full_h = h as u32;
        }
        if (w as u64).saturating_mul(h as u64).saturating_mul(4) > max_decode_bytes {
            bail!("HEIC/HEIF image too large ({w}x{h})");
        }

        let cs = CgColorSpace(unsafe { CGColorSpaceCreateWithName(kCGColorSpaceSRGB) });
        if cs.0.is_null() {
            bail!("CoreGraphics failed to create sRGB colour space");
        }
        let mut rgba = vec![0u8; w * h * 4];
        let ctx = CgContext(unsafe {
            CGBitmapContextCreate(
                rgba.as_mut_ptr() as *mut c_void,
                w,
                h,
                8,
                w * 4,
                cs.0,
                K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST | K_CG_BITMAP_BYTE_ORDER_32_BIG,
            )
        });
        if ctx.0.is_null() {
            bail!("CoreGraphics failed to create bitmap context");
        }
        let rect = CGRect { x: 0.0, y: 0.0, w: w as f64, h: h as f64 };
        unsafe {
            CGContextSetRGBFillColor(ctx.0, 1.0, 1.0, 1.0, 1.0);
            CGContextFillRect(ctx.0, rect);
            CGContextDrawImage(ctx.0, rect, cg_img.0);
        }
        drop(ctx);

        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        let image = RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| anyhow!("HEIC/HEIF RGB buffer"))?;
        Ok(DecodedHeif {
            image,
            full_width: full_w,
            full_height: full_h,
            icc: None,
            already_srgb: true,
            had_alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_heic_brands_and_ignores_cr3_and_jpeg() {
        let mut heic = Vec::new();
        heic.extend_from_slice(&24u32.to_be_bytes());
        heic.extend_from_slice(b"ftypheic\0\0\0\0mif1heic");
        assert!(is_heif_container(&heic));

        let mut mif1_compat = Vec::new();
        mif1_compat.extend_from_slice(&28u32.to_be_bytes());
        mif1_compat.extend_from_slice(b"ftypmp41\0\0\0\0isomheicmif1");
        assert!(is_heif_container(&mif1_compat));

        let mut cr3 = Vec::new();
        cr3.extend_from_slice(&24u32.to_be_bytes());
        cr3.extend_from_slice(b"ftypcrx \0\0\0\0crx isom");
        assert!(!is_heif_container(&cr3));

        assert!(!is_heif_container(&[0xFF, 0xD8, 0xFF, 0xE0]));
    }
}
