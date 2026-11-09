//! The set of distinct source photos available for one mosaic.
//!
//! Identity is the content hash, so a photo that appears in several selected
//! libraries (or several times in one) is a single pool entry. Optionally,
//! near-duplicates (same 64-bit dHash, same aspect, nearly identical colour)
//! are merged into one *group*; reuse limits and repeat penalties then apply
//! to the group, so resized copies cannot inflate variety either.

use crate::catalog::{Catalog, Hash, PoolRow};
use crate::color::{de2, Lab};
use crate::descriptor::{crop_descriptor, mean_lab};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct PoolPhoto {
    pub hash: Hash,
    pub width: u32,
    pub height: u32,
    pub grid: Vec<u8>,
    pub dhash: u64,
    pub mean: Lab,
    /// Duplicate group (index into groups); equal for merged near-duplicates.
    pub group: u32,
    /// Faces in the upright frame, used to place crops.
    pub faces: Vec<crate::faces::Face>,
}

impl PoolPhoto {
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

#[derive(Debug, Clone, Default)]
pub struct Pool {
    pub photos: Vec<PoolPhoto>,
    pub group_count: u32,
}

impl Pool {
    pub fn load(cat: &Catalog, libs: &[i64], merge_near_duplicates: bool) -> Result<Self> {
        let rows = cat.pool_rows(libs)?;
        if rows.is_empty() {
            bail!("the selected libraries contain no indexed photos; add folders and update the index first");
        }
        Ok(Self::from_rows(rows, merge_near_duplicates))
    }

    pub fn from_rows(rows: Vec<PoolRow>, merge_near_duplicates: bool) -> Self {
        let mut seen = HashSet::new();
        let photos: Vec<PoolPhoto> = rows
            .into_iter()
            .filter(|r| seen.insert(r.hash))
            .map(|r| {
                let mean = mean_lab(&crop_descriptor(&r.grid, [0.0, 0.0, 1.0, 1.0]));
                PoolPhoto { hash: r.hash, width: r.width, height: r.height, grid: r.grid, dhash: r.dhash, mean, group: 0, faces: r.faces }
            })
            .collect();
        let mut p = Pool { photos, group_count: 0 };
        p.assign_groups(merge_near_duplicates);
        p
    }

    fn assign_groups(&mut self, merge: bool) {
        let n = self.photos.len();
        let mut group = vec![u32::MAX; n];
        let mut next = 0u32;
        if merge {
            let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
            for (i, p) in self.photos.iter().enumerate() {
                by_hash.entry(p.dhash).or_default().push(i);
            }
            let mut keys: Vec<u64> = by_hash.keys().copied().collect();
            keys.sort_unstable();
            for k in keys {
                let members = &by_hash[&k];
                for (a_pos, &a) in members.iter().enumerate() {
                    if group[a] != u32::MAX {
                        continue;
                    }
                    group[a] = next;
                    for &b in &members[a_pos + 1..] {
                        if group[b] == u32::MAX && near_duplicate(&self.photos[a], &self.photos[b]) {
                            group[b] = next;
                        }
                    }
                    next += 1;
                }
            }
        } else {
            for g in group.iter_mut() {
                *g = next;
                next += 1;
            }
        }
        for (p, g) in self.photos.iter_mut().zip(group) {
            p.group = g;
        }
        self.group_count = next;
    }

    /// A copy without the given photos and anything in the same duplicate group
    /// or with a near-identical dHash (used to evaluate a library photo as motif).
    pub fn excluding(&self, hash: &Hash, dhash: Option<u64>, merge: bool) -> Pool {
        let excl_groups: HashSet<u32> = self.photos.iter().filter(|p| &p.hash == hash).map(|p| p.group).collect();
        let rows: Vec<PoolRow> = self
            .photos
            .iter()
            .filter(|p| &p.hash != hash && !excl_groups.contains(&p.group))
            .filter(|p| dhash.map(|d| crate::descriptor::hamming(d, p.dhash) > 4).unwrap_or(true))
            .map(|p| PoolRow { hash: p.hash, width: p.width, height: p.height, grid: p.grid.clone(), dhash: p.dhash, faces: p.faces.clone() })
            .collect();
        Pool::from_rows(rows, merge)
    }

    pub fn index_of(&self, h: &Hash) -> Option<usize> {
        self.photos.iter().position(|p| &p.hash == h)
    }
}

fn near_duplicate(a: &PoolPhoto, b: &PoolPhoto) -> bool {
    (a.aspect() / b.aspect() - 1.0).abs() < 0.02 && de2(&a.mean, &b.mean) < 9.0
}
