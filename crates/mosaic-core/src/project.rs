//! Mosaic projects: everything needed to reproduce and re-render a mosaic.
//! Projects are JSON files stored wherever the user chooses, separate from
//! the catalogue (cached analysis) and from the source photos.

use crate::atomic::TempOutput;
use crate::catalog::{hex, unhex, Catalog, Hash};
use crate::color::{dequantize_lab, quantize_lab, Lab};
use crate::decode;
use crate::descriptor::{CELLS, DESCRIPTOR_VERSION};
use crate::job::Job;
use crate::matching::{match_tiles, GridSpec, MatchError, MatchSettings, MatchStats, Target, TileAssign};
use crate::pool::Pool;
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const PROJECT_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifRef {
    pub path: PathBuf,
    pub content_hash: String,
    pub width: u32,
    pub height: u32,
    /// Normalised part of the motif that the mosaic covers (cover fit, centred).
    pub crop: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryRef {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub hash: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub format: u32,
    pub app_version: String,
    pub motif: MotifRef,
    pub libraries: Vec<LibraryRef>,
    pub grid: GridSpec,
    pub settings: MatchSettings,
    pub descriptor_version: u32,
    pub sources: Vec<SourceRef>,
    /// Row-major tile assignments; `photo` indexes `sources`.
    pub tiles: Vec<TileAssign>,
    /// Target motif cells (CELLS×CELLS per tile), quantised Lab, base64.
    pub target_cells: String,
    pub stats: MatchStats,
}

/// Load the motif and sample it for the grid.
pub fn load_target(motif: &Path, grid: &GridSpec) -> Result<(Target, MotifRef)> {
    let bytes = std::fs::read(motif).with_context(|| format!("cannot read motif {}", motif.display()))?;
    let need = (grid.cols.max(grid.rows) as usize * CELLS * 2) as u32;
    let d = decode::decode_bytes(&bytes, Some(need)).with_context(|| format!("cannot decode motif {}", motif.display()))?;
    let (target, crop) = Target::from_image(&d.image, grid);
    let mref = MotifRef {
        path: std::fs::canonicalize(motif).unwrap_or_else(|_| motif.to_path_buf()),
        content_hash: hex(blake3::hash(&bytes).as_bytes()),
        width: d.info.full_width,
        height: d.info.full_height,
        crop,
    };
    Ok((target, mref))
}

impl Project {
    /// Match and build a project from an already loaded pool.
    pub fn create(
        pool: &Pool,
        libraries: Vec<LibraryRef>,
        target: &Target,
        motif: MotifRef,
        grid: GridSpec,
        settings: MatchSettings,
        job: &Job,
    ) -> Result<Project, MatchError> {
        let (tiles, stats) = match_tiles(pool, target, &grid, &settings, job)?;
        // Keep only the photos actually used, in first-use order.
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut sources = Vec::new();
        let tiles = tiles
            .into_iter()
            .map(|mut t| {
                let next = remap.len() as u32;
                let idx = *remap.entry(t.photo).or_insert_with(|| {
                    let p = &pool.photos[t.photo as usize];
                    sources.push(SourceRef { hash: hex(&p.hash), width: p.width, height: p.height });
                    next
                });
                t.photo = idx;
                t
            })
            .collect();
        Ok(Project {
            format: PROJECT_FORMAT,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            motif,
            libraries,
            grid,
            settings,
            descriptor_version: DESCRIPTOR_VERSION,
            sources,
            tiles,
            target_cells: encode_cells(&target.cells()),
            stats,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out = TempOutput::create(path)?;
        serde_json::to_writer(std::io::BufWriter::new(out.file()), self)?;
        out.file().flush()?;
        out.commit()
    }

    pub fn load(path: &Path) -> Result<Project> {
        let f = std::fs::File::open(path).with_context(|| format!("cannot open project {}", path.display()))?;
        let p: Project = serde_json::from_reader(std::io::BufReader::new(f)).with_context(|| format!("invalid project file {}", path.display()))?;
        if p.format != PROJECT_FORMAT {
            bail!("unsupported project format {}", p.format);
        }
        if p.tiles.len() != p.grid.tiles() {
            bail!("project is inconsistent: {} tiles for a {}×{} grid", p.tiles.len(), p.grid.cols, p.grid.rows);
        }
        if p.tiles.iter().any(|t| t.photo as usize >= p.sources.len()) {
            bail!("project is inconsistent: tile refers to unknown source");
        }
        Ok(p)
    }

    pub fn target_cells(&self) -> Result<Vec<Lab>> {
        decode_cells(&self.target_cells, self.grid.cols as usize * CELLS * self.grid.rows as usize * CELLS)
    }

    pub fn source_hashes(&self) -> Result<Vec<Hash>> {
        self.sources.iter().map(|s| unhex(&s.hash)).collect()
    }

    pub fn library_ids(&self) -> Vec<i64> {
        self.libraries.iter().map(|l| l.id).collect()
    }

    /// Uses per source index.
    pub fn usage(&self) -> Vec<u32> {
        let mut u = vec![0u32; self.sources.len()];
        for t in &self.tiles {
            u[t.photo as usize] += 1;
        }
        u
    }
}

fn encode_cells(cells: &[Lab]) -> String {
    let mut b = Vec::with_capacity(cells.len() * 3);
    for c in cells {
        b.extend_from_slice(&quantize_lab(*c));
    }
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn decode_cells(s: &str, n: usize) -> Result<Vec<Lab>> {
    let b = base64::engine::general_purpose::STANDARD.decode(s)?;
    if b.len() != n * 3 {
        bail!("target cells have wrong size");
    }
    Ok(b.chunks_exact(3).map(|c| dequantize_lab([c[0], c[1], c[2]])).collect())
}

// ---------------------------------------------------------------------------
// Source validation before export.

#[derive(Debug, Clone, Serialize)]
pub enum ProblemKind {
    /// No file with this content exists in the project's libraries.
    NotIndexed,
    /// Indexed file(s) are missing on disk.
    Missing,
    /// Indexed file(s) exist but their content changed.
    Changed,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceProblem {
    pub source: usize,
    pub hash: String,
    pub kind: ProblemKind,
    pub paths: Vec<PathBuf>,
    pub tiles: u32,
}

#[derive(Debug, Clone, Default)]
pub struct SourceCheck {
    pub resolved: Vec<Option<PathBuf>>,
    pub problems: Vec<SourceProblem>,
    pub rehashed: usize,
}

impl SourceCheck {
    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
    pub fn report(&self, p: &Project) -> String {
        if self.ok() {
            return format!("All {} source photos are present and unchanged.", p.sources.len());
        }
        let tiles: u32 = self.problems.iter().map(|x| x.tiles).sum();
        let mut s = format!("{} of {} source photos are unavailable ({} tiles affected):\n", self.problems.len(), p.sources.len(), tiles);
        for pr in self.problems.iter().take(20) {
            let where_ = pr.paths.first().map(|p| p.display().to_string()).unwrap_or_else(|| "(no indexed file)".into());
            s += &format!("  {:?}: {} [{} tiles] {}\n", pr.kind, &pr.hash[..12], pr.tiles, where_);
        }
        if self.problems.len() > 20 {
            s += &format!("  … and {} more\n", self.problems.len() - 20);
        }
        s += "Fix: reconnect or relink the source folder (library relink), restore the files, or update the library index.\n";
        s += "Re-matching the project would choose different photos; it is never done automatically.";
        s
    }
}

/// Resolve each source to an existing, unchanged file. Never substitutes.
pub fn check_sources(cat: &Catalog, p: &Project, job: &Job) -> Result<SourceCheck> {
    let libs = p.library_ids();
    let usage = p.usage();
    let mut out = SourceCheck::default();
    for (i, s) in p.sources.iter().enumerate() {
        if i % 256 == 0 {
            job.check()?;
            job.report("checking source photos", i as u64, p.sources.len() as u64);
        }
        let h = unhex(&s.hash)?;
        let locs = cat.locations(&h, &libs)?;
        let mut found = None;
        let mut kind = if locs.is_empty() { ProblemKind::NotIndexed } else { ProblemKind::Missing };
        for l in &locs {
            let Ok(m) = std::fs::metadata(&l.path) else { continue };
            let mtime = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as i64).unwrap_or(0);
            if m.len() as i64 == l.size && mtime == l.mtime_ns {
                found = Some(l.path.clone());
                break;
            }
            // Stat differs: verify content.
            out.rehashed += 1;
            match std::fs::read(&l.path) {
                Ok(b) if blake3::hash(&b).as_bytes() == &h => {
                    found = Some(l.path.clone());
                    break;
                }
                Ok(_) => kind = ProblemKind::Changed,
                Err(_) => {}
            }
        }
        if found.is_none() {
            out.problems.push(SourceProblem { source: i, hash: s.hash.clone(), kind, paths: locs.iter().map(|l| l.path.clone()).collect(), tiles: usage[i] });
        }
        out.resolved.push(found);
    }
    Ok(out)
}

/// CSV manifest of photos actually used, with supplied attribution metadata.
pub fn write_manifest(cat: &Catalog, p: &Project, out: &Path) -> Result<usize> {
    use crate::indexer::csv_field;
    let libs = p.library_ids();
    let usage = p.usage();
    let mut t = TempOutput::create(out)?;
    {
        let mut w = std::io::BufWriter::new(t.file());
        writeln!(w, "content_hash,uses,path,library,artist,copyright,license,source,width,height")?;
        for (i, s) in p.sources.iter().enumerate() {
            let h = unhex(&s.hash)?;
            let locs = cat.locations(&h, &libs)?;
            let l = locs.first();
            let a = l.map(|l| l.attribution.clone()).unwrap_or_default();
            let f = |o: &Option<String>| csv_field(o.as_deref().unwrap_or(""));
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{}",
                s.hash,
                usage[i],
                csv_field(&l.map(|l| l.path.display().to_string()).unwrap_or_default()),
                csv_field(&l.map(|l| l.library.clone()).unwrap_or_default()),
                f(&a.artist),
                f(&a.copyright),
                f(&a.license),
                f(&a.source),
                s.width,
                s.height
            )?;
        }
        w.flush()?;
    }
    t.commit()?;
    Ok(p.sources.len())
}

pub fn libraries_refs(cat: &Catalog, ids: &[i64]) -> Result<Vec<LibraryRef>> {
    ids.iter().map(|&id| Ok(LibraryRef { id, name: cat.library_name(id).map_err(|_| anyhow!("library {id} not found"))? })).collect()
}
