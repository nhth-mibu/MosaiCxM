//! Write-to-temporary-then-rename outputs. A cancelled or failed job never
//! leaves a file at the final path; the job-owned temporary file is removed.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct TempOutput {
    final_path: PathBuf,
    temp_path: PathBuf,
    file: Option<File>,
    committed: bool,
}

pub const TEMP_MARKER: &str = ".mosaic-partial-";

impl TempOutput {
    pub fn create(final_path: &Path) -> Result<Self> {
        let dir = match final_path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
            _ => PathBuf::from("."),
        };
        if !dir.is_dir() {
            bail!("output folder does not exist: {}", dir.display());
        }
        let name = final_path.file_name().and_then(|n| n.to_str()).unwrap_or("output");
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let temp_path = dir.join(format!(".{name}{TEMP_MARKER}{}-{nonce}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| format!("cannot create {}", temp_path.display()))?;
        Ok(Self { final_path: final_path.to_path_buf(), temp_path, file: Some(file), committed: false })
    }

    pub fn file(&mut self) -> &mut File {
        self.file.as_mut().expect("file open")
    }

    pub fn temp_path(&self) -> &Path {
        &self.temp_path
    }

    /// Flush to disk and atomically move to the final path.
    pub fn commit(mut self) -> Result<()> {
        let f = self.file.take().expect("file open");
        f.sync_all().context("flushing output to disk")?;
        drop(f);
        std::fs::rename(&self.temp_path, &self.final_path)
            .with_context(|| format!("cannot move output into place at {}", self.final_path.display()))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if !self.committed {
            self.file.take();
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

/// Fails early when the output folder obviously lacks space.
pub fn ensure_disk_space(final_path: &Path, needed: u64) -> Result<()> {
    let dir = final_path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if let Ok(avail) = fs4::available_space(dir) {
        if avail < needed {
            bail!(
                "not enough disk space in {}: about {} needed, {} available",
                dir.display(),
                crate::units::format_bytes(needed),
                crate::units::format_bytes(avail)
            );
        }
    }
    Ok(())
}
