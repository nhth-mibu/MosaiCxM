//! Photo mosaic engine: libraries, indexing, matching, rendering and export.
//! Shared by the `mosaic` CLI and the desktop GUI.

pub mod atomic;
pub mod catalog;
pub mod color;
pub mod decode;
pub mod descriptor;
pub mod indexer;
pub mod job;
pub mod matching;
pub mod pool;
pub mod project;
pub mod render;
pub mod rng;
pub mod shape;
pub mod units;

pub use catalog::Catalog;
pub use job::Job;
pub use matching::{GridSpec, MatchSettings, Method};
pub use project::Project;
pub mod export;
pub mod layout;
pub mod pdf;
pub mod tiff;
pub mod recommend;

/// Configure the global worker pool (decoding, matching and rendering share
/// it, which avoids oversubscription). Call once at start-up.
pub fn init_threads(n: usize) -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).build_global().map_err(|e| anyhow::anyhow!("{e}"))
}
