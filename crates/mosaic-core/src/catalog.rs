//! Persistent catalogue (SQLite) of libraries, source roots, files,
//! descriptors and thumbnails. Original photos are never modified; removing a
//! library or folder only deletes catalogue rows.

use crate::descriptor::{Descriptor, DESCRIPTOR_VERSION, GRID_BYTES};
use crate::faces::{self, Face, FACES_VERSION};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type Hash = [u8; 32];

pub fn hex(h: &Hash) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn unhex(s: &str) -> Result<Hash> {
    if s.len() != 64 {
        bail!("bad hash {s}");
    }
    let mut h = [0u8; 32];
    for i in 0..32 {
        h[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }
    Ok(h)
}

pub const STATUS_OK: i64 = 0;
pub const STATUS_SKIPPED: i64 = 1;

#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub id: i64,
    pub name: String,
    pub roots: Vec<RootInfo>,
    pub files_ok: u64,
    pub files_skipped: u64,
    pub distinct_photos: u64,
}

#[derive(Debug, Clone)]
pub struct RootInfo {
    pub id: i64,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct FileRec {
    pub id: i64,
    pub size: i64,
    pub mtime_ns: i64,
    pub hash: Option<Hash>,
    pub status: i64,
    pub meta_version: i64,
}

#[derive(Debug, Clone, Default)]
pub struct Attribution {
    pub artist: Option<String>,
    pub copyright: Option<String>,
    pub license: Option<String>,
    pub source: Option<String>,
}

/// Camera metadata read from EXIF (all optional; many images have none).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PhotoExif {
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub f_number: Option<f64>,
    pub exposure_s: Option<f64>,
    pub iso: Option<i64>,
    pub focal_mm: Option<f64>,
    pub taken: Option<String>,
}

impl PhotoExif {
    pub fn is_empty(&self) -> bool {
        self.camera.is_none() && self.lens.is_none() && self.f_number.is_none() && self.exposure_s.is_none() && self.iso.is_none() && self.focal_mm.is_none() && self.taken.is_none()
    }
}

#[derive(Debug, Clone)]
pub struct Location {
    pub library: String,
    pub path: PathBuf,
    pub size: i64,
    pub mtime_ns: i64,
    pub attribution: Attribution,
    pub exif: PhotoExif,
}

/// Row used to build a matching pool.
pub struct PoolRow {
    pub hash: Hash,
    pub width: u32,
    pub height: u32,
    pub grid: Vec<u8>,
    pub dhash: u64,
    /// Faces found while indexing (empty when none, or not yet detected).
    pub faces: Vec<Face>,
}

pub struct Catalog {
    pub conn: Connection,
    pub path: PathBuf,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS library(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS root(
  id INTEGER PRIMARY KEY,
  library_id INTEGER NOT NULL REFERENCES library(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  UNIQUE(library_id, path));
CREATE TABLE IF NOT EXISTS file(
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES root(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  content_hash BLOB,
  status INTEGER NOT NULL,
  error TEXT,
  artist TEXT, copyright TEXT, license TEXT, source TEXT,
  UNIQUE(root_id, rel_path));
CREATE INDEX IF NOT EXISTS file_hash ON file(content_hash);
CREATE TABLE IF NOT EXISTS descriptor(
  content_hash BLOB PRIMARY KEY,
  version INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  grid BLOB NOT NULL,
  dhash INTEGER NOT NULL,
  has_icc INTEGER NOT NULL,
  had_alpha INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS thumb(content_hash BLOB PRIMARY KEY, version INTEGER NOT NULL, jpeg BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS faces(content_hash BLOB PRIMARY KEY, version INTEGER NOT NULL, boxes BLOB NOT NULL);
"#;

impl Catalog {
    pub fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir).with_context(|| format!("create {}", data_dir.display()))?;
        let path = data_dir.join("catalog.sqlite");
        let conn = Connection::open(&path).with_context(|| format!("open {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        // Additive migrations (ignore "duplicate column" on already-migrated catalogues).
        for col in ["camera TEXT", "lens TEXT", "f_number REAL", "exposure_s REAL", "iso INTEGER", "focal_mm REAL", "taken TEXT", "meta_version INTEGER NOT NULL DEFAULT 0"] {
            let _ = conn.execute(&format!("ALTER TABLE file ADD COLUMN {col}"), []);
        }
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1')", [])?;
        Ok(Self { conn, path })
    }

    pub fn create_library(&self, name: &str) -> Result<i64> {
        let name = name.trim();
        if name.is_empty() {
            bail!("library name must not be empty");
        }
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() as i64;
        self.conn
            .execute("INSERT INTO library(name, created_at) VALUES(?1, ?2)", params![name, now])
            .map_err(|e| anyhow!("cannot create library '{name}': {e}"))?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn library_id(&self, name: &str) -> Result<i64> {
        self.conn
            .query_row("SELECT id FROM library WHERE name = ?1", [name], |r| r.get(0))
            .optional()?
            .ok_or_else(|| anyhow!("no library named '{name}'"))
    }

    pub fn library_name(&self, id: i64) -> Result<String> {
        Ok(self.conn.query_row("SELECT name FROM library WHERE id = ?1", [id], |r| r.get(0))?)
    }

    pub fn rename_library(&self, id: i64, new_name: &str) -> Result<()> {
        self.conn.execute("UPDATE library SET name = ?1 WHERE id = ?2", params![new_name.trim(), id])?;
        Ok(())
    }

    /// Removes the library from the catalogue. Photos on disk are untouched.
    pub fn remove_library(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM library WHERE id = ?1", [id])?;
        Ok(())
    }

    pub fn libraries(&self) -> Result<Vec<LibraryInfo>> {
        let mut st = self.conn.prepare("SELECT id, name FROM library ORDER BY name")?;
        let libs: Vec<(i64, String)> = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        libs.into_iter()
            .map(|(id, name)| {
                let roots = self.roots(id)?;
                let (ok, skipped, distinct): (i64, i64, i64) = self.conn.query_row(
                    "SELECT COALESCE(SUM(f.status = 0), 0), COALESCE(SUM(f.status = 1), 0),
                            COUNT(DISTINCT CASE WHEN f.status = 0 THEN f.content_hash END)
                     FROM file f JOIN root r ON f.root_id = r.id WHERE r.library_id = ?1",
                    [id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
                Ok(LibraryInfo { id, name, roots, files_ok: ok as u64, files_skipped: skipped as u64, distinct_photos: distinct as u64 })
            })
            .collect()
    }

    pub fn roots(&self, lib: i64) -> Result<Vec<RootInfo>> {
        let mut st = self.conn.prepare("SELECT id, path FROM root WHERE library_id = ?1 ORDER BY path")?;
        let v = st
            .query_map([lib], |r| Ok(RootInfo { id: r.get(0)?, path: PathBuf::from(r.get::<_, String>(1)?) }))?
            .collect::<Result<_, _>>()?;
        Ok(v)
    }

    pub fn add_root(&self, lib: i64, path: &Path) -> Result<i64> {
        let canon = std::fs::canonicalize(path).with_context(|| format!("folder not found: {}", path.display()))?;
        if !canon.is_dir() {
            bail!("not a folder: {}", canon.display());
        }
        let s = path_str(&canon)?;
        for r in self.roots(lib)? {
            if canon.starts_with(&r.path) || r.path.starts_with(&canon) {
                bail!("{} overlaps existing source folder {}", canon.display(), r.path.display());
            }
        }
        self.conn.execute("INSERT INTO root(library_id, path) VALUES(?1, ?2)", params![lib, s])?;
        Ok(self.conn.last_insert_rowid())
    }

    fn find_root(&self, lib: i64, path: &Path) -> Result<RootInfo> {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.roots(lib)?
            .into_iter()
            .find(|r| r.path == canon || r.path == path)
            .ok_or_else(|| anyhow!("{} is not a source folder of this library", path.display()))
    }

    pub fn remove_root(&self, lib: i64, path: &Path) -> Result<()> {
        let r = self.find_root(lib, path)?;
        self.conn.execute("DELETE FROM root WHERE id = ?1", [r.id])?;
        Ok(())
    }

    /// Points an existing source folder at a new location. Unchanged files keep
    /// their records and descriptors. Verifies a sample of files unless `force`.
    pub fn relink_root(&self, lib: i64, old: &Path, new: &Path, force: bool) -> Result<RelinkReport> {
        let r = self.find_root(lib, old)?;
        let canon = std::fs::canonicalize(new).with_context(|| format!("new folder not found: {}", new.display()))?;
        let mut st = self.conn.prepare("SELECT rel_path, size FROM file WHERE root_id = ?1 AND status = 0 ORDER BY id LIMIT 50")?;
        let sample: Vec<(String, i64)> = st.query_map([r.id], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<Result<_, _>>()?;
        let found = sample
            .iter()
            .filter(|(rel, size)| std::fs::metadata(canon.join(rel)).map(|m| m.len() as i64 == *size).unwrap_or(false))
            .count();
        let report = RelinkReport { sampled: sample.len(), found };
        if !force && !sample.is_empty() && found * 2 < sample.len() {
            bail!(
                "only {found} of {} sampled files exist at {} with the same size; use --force to relink anyway",
                sample.len(),
                canon.display()
            );
        }
        self.conn.execute("UPDATE root SET path = ?1 WHERE id = ?2", params![path_str(&canon)?, r.id])?;
        Ok(report)
    }

    pub fn files_of_root(&self, root: i64) -> Result<HashMap<String, FileRec>> {
        let mut st = self.conn.prepare("SELECT id, rel_path, size, mtime_ns, content_hash, status, meta_version FROM file WHERE root_id = ?1")?;
        let rows = st.query_map([root], |r| {
            let h: Option<Vec<u8>> = r.get(4)?;
            Ok((
                r.get::<_, String>(1)?,
                FileRec {
                    id: r.get(0)?,
                    size: r.get(2)?,
                    mtime_ns: r.get(3)?,
                    hash: h.and_then(|v| v.try_into().ok()),
                    status: r.get(5)?,
                    meta_version: r.get(6)?,
                },
            ))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn has_current_descriptor(&self, h: &Hash) -> Result<bool> {
        Ok(self
            .conn
            .query_row("SELECT 1 FROM descriptor WHERE content_hash = ?1 AND version = ?2", params![&h[..], DESCRIPTOR_VERSION], |_| Ok(()))
            .optional()?
            .is_some())
    }

    pub fn put_descriptor(&self, h: &Hash, d: &Descriptor, has_icc: bool, had_alpha: bool, thumb: &[u8]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO descriptor(content_hash, version, width, height, grid, dhash, has_icc, had_alpha)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![&h[..], DESCRIPTOR_VERSION, d.width, d.height, &d.grid, d.dhash as i64, has_icc, had_alpha],
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO thumb(content_hash, version, jpeg) VALUES(?1, ?2, ?3)",
            params![&h[..], DESCRIPTOR_VERSION, thumb],
        )?;
        Ok(())
    }

    pub fn put_faces(&self, h: &Hash, f: &[Face]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO faces(content_hash, version, boxes) VALUES(?1, ?2, ?3)",
            params![&h[..], FACES_VERSION, faces::encode(f)],
        )?;
        Ok(())
    }

    pub fn thumb(&self, h: &Hash) -> Result<Option<Vec<u8>>> {
        Ok(self.conn.query_row("SELECT jpeg FROM thumb WHERE content_hash = ?1", [&h[..]], |r| r.get(0)).optional()?)
    }

    pub fn descriptor(&self, h: &Hash) -> Result<Option<PoolRow>> {
        Ok(self
            .conn
            .query_row(
                "SELECT d.width, d.height, d.grid, d.dhash, fc.boxes FROM descriptor d
                 LEFT JOIN faces fc ON fc.content_hash = d.content_hash AND fc.version = ?3
                 WHERE d.content_hash = ?1 AND d.version = ?2",
                params![&h[..], DESCRIPTOR_VERSION, FACES_VERSION],
                |r| {
                    let boxes: Option<Vec<u8>> = r.get(4)?;
                    Ok(PoolRow { hash: *h, width: r.get(0)?, height: r.get(1)?, grid: r.get(2)?, dhash: r.get::<_, i64>(3)? as u64, faces: boxes.map(|b| faces::decode(&b)).unwrap_or_default() })
                },
            )
            .optional()?)
    }

    /// Distinct photos (by content) across the given libraries.
    pub fn pool_rows(&self, libs: &[i64]) -> Result<Vec<PoolRow>> {
        let ids = libs.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT d.content_hash, d.width, d.height, d.grid, d.dhash, fc.boxes FROM descriptor d
             LEFT JOIN faces fc ON fc.content_hash = d.content_hash AND fc.version = {FACES_VERSION}
             WHERE d.version = {DESCRIPTOR_VERSION} AND d.content_hash IN (
               SELECT f.content_hash FROM file f JOIN root r ON f.root_id = r.id
               WHERE f.status = 0 AND r.library_id IN ({ids}))
             ORDER BY d.content_hash"
        );
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map([], |r| {
            let h: Vec<u8> = r.get(0)?;
            Ok(PoolRow {
                hash: h.try_into().unwrap_or([0; 32]),
                width: r.get(1)?,
                height: r.get(2)?,
                grid: r.get(3)?,
                dhash: r.get::<_, i64>(4)? as u64,
                faces: r.get::<_, Option<Vec<u8>>>(5)?.map(|b| faces::decode(&b)).unwrap_or_default(),
            })
        })?;
        let v: Vec<PoolRow> = rows.collect::<Result<_, _>>()?;
        Ok(v.into_iter().filter(|r| r.grid.len() == GRID_BYTES).collect())
    }

    /// All known file locations of a content hash inside the given libraries.
    pub fn locations(&self, h: &Hash, libs: &[i64]) -> Result<Vec<Location>> {
        let ids = libs.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT l.name, r.path, f.rel_path, f.size, f.mtime_ns, f.artist, f.copyright, f.license, f.source,
                    f.camera, f.lens, f.f_number, f.exposure_s, f.iso, f.focal_mm, f.taken
             FROM file f JOIN root r ON f.root_id = r.id JOIN library l ON r.library_id = l.id
             WHERE f.content_hash = ?1 AND f.status = 0 AND r.library_id IN ({ids}) ORDER BY l.name, f.rel_path"
        );
        let mut st = self.conn.prepare_cached(&sql)?;
        let rows = st.query_map([&h[..]], |r| {
            Ok(Location {
                library: r.get(0)?,
                path: PathBuf::from(r.get::<_, String>(1)?).join(r.get::<_, String>(2)?),
                size: r.get(3)?,
                mtime_ns: r.get(4)?,
                attribution: Attribution { artist: r.get(5)?, copyright: r.get(6)?, license: r.get(7)?, source: r.get(8)? },
                exif: PhotoExif { camera: r.get(9)?, lens: r.get(10)?, f_number: r.get(11)?, exposure_s: r.get(12)?, iso: r.get(13)?, focal_mm: r.get(14)?, taken: r.get(15)? },
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn skipped(&self, lib: i64) -> Result<Vec<(PathBuf, String)>> {
        let mut st = self.conn.prepare(
            "SELECT r.path, f.rel_path, COALESCE(f.error, '') FROM file f JOIN root r ON f.root_id = r.id
             WHERE r.library_id = ?1 AND f.status = 1 ORDER BY r.path, f.rel_path",
        )?;
        let v = st
            .query_map([lib], |r| Ok((PathBuf::from(r.get::<_, String>(0)?).join(r.get::<_, String>(1)?), r.get(2)?)))?
            .collect::<Result<_, _>>()?;
        Ok(v)
    }

    /// Deletes cached descriptors and thumbnails no file refers to.
    pub fn gc(&self) -> Result<usize> {
        let n = self.conn.execute("DELETE FROM descriptor WHERE content_hash NOT IN (SELECT content_hash FROM file WHERE content_hash IS NOT NULL)", [])?;
        self.conn.execute("DELETE FROM thumb WHERE content_hash NOT IN (SELECT content_hash FROM descriptor)", [])?;
        self.conn.execute("DELETE FROM faces WHERE content_hash NOT IN (SELECT content_hash FROM descriptor)", [])?;
        Ok(n)
    }
}

#[derive(Debug, Clone)]
pub struct RelinkReport {
    pub sampled: usize,
    pub found: usize,
}

pub fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

pub fn library_ids(cat: &Catalog, names: &[String]) -> Result<Vec<i64>> {
    if names.is_empty() {
        bail!("select at least one library");
    }
    names.iter().map(|n| cat.library_id(n)).collect()
}
