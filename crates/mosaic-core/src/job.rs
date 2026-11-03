//! Cancellation and progress reporting shared by all long-running engine jobs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct Progress {
    pub stage: String,
    pub done: u64,
    pub total: u64,
}

/// Error returned when a job was cancelled. Detect with `is_cancelled`.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

type ProgressFn = dyn Fn(&Progress) + Send + Sync;

#[derive(Clone)]
pub struct Job {
    cancel: Arc<AtomicBool>,
    progress: Arc<ProgressFn>,
}

impl Default for Job {
    fn default() -> Self {
        Self::new(|_| {})
    }
}

impl Job {
    pub fn new(progress: impl Fn(&Progress) + Send + Sync + 'static) -> Self {
        Self { cancel: Arc::new(AtomicBool::new(false)), progress: Arc::new(progress) }
    }
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = flag;
        self
    }
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    /// Returns `Err(Cancelled)` if cancellation was requested.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.is_cancelled() { Err(Cancelled.into()) } else { Ok(()) }
    }
    pub fn report(&self, stage: &str, done: u64, total: u64) {
        (self.progress)(&Progress { stage: stage.to_string(), done, total });
    }
}
