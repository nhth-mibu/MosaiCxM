//! Update check: asks GitHub for the latest release and compares it with the
//! version baked into this build. It only tells the user; it never downloads
//! or installs anything.
//!
//! Release builds get their version from the `MOSAIC_VERSION` environment
//! variable at compile time (set by the Release workflow, like `v0.7.3`).
//! Local and test builds have no version: they skip the check at start.

use anyhow::{anyhow, Result};
use std::path::Path;
use std::time::Duration;

const LATEST_URL: &str = "https://api.github.com/repos/nhth-mibu/MosaiCxM/releases/latest";
const PREFS: &str = "update-prefs.json";

/// The version of this build without the leading `v`, or `None` for a
/// development build.
pub fn current_version() -> Option<&'static str> {
    let v = option_env!("MOSAIC_VERSION")?;
    let v = v.strip_prefix('v').unwrap_or(v);
    parse(v).map(|_| v)
}

/// `1.2.3` or `1.2.3-beta.1` → (1, 2, 3, is_prerelease).
fn parse(v: &str) -> Option<(u64, u64, u64, bool)> {
    let v = v.trim().trim_start_matches('v');
    let (core, pre) = match v.split_once(['-', '+']) {
        Some((c, _)) => (c, v.as_bytes()[c.len()] == b'-'),
        None => (v, false),
    };
    let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
    let (a, b, c) = (it.next()??, it.next()??, it.next()??);
    if it.next().is_some() {
        return None;
    }
    Some((a, b, c, pre))
}

/// True when `latest` is a newer version than `current`.
pub fn is_newer(latest: &str, current: &str) -> bool {
    match (parse(latest), parse(current)) {
        (Some((a, b, c, lp)), Some((x, y, z, cp))) => (a, b, c) > (x, y, z) || ((a, b, c) == (x, y, z) && cp && !lp),
        _ => false,
    }
}

pub struct Release {
    /// Version without the leading `v`.
    pub version: String,
}

/// The GitHub page of one release, where the downloads are.
pub fn release_page(version: &str) -> String {
    format!("https://github.com/nhth-mibu/MosaiCxM/releases/tag/v{version}")
}

/// Asks GitHub for the latest published release (drafts and pre-releases
/// are not included). Takes at most about ten seconds.
pub fn fetch_latest() -> Result<Release> {
    // Use the system's certificates, so the check also works behind company
    // proxies that the system trusts.
    let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build();
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(10))).tls_config(tls).build().into();
    let body = agent
        .get(LATEST_URL)
        .header("User-Agent", "MosaiCxM-update-check")
        .header("Accept", "application/vnd.github+json")
        .call()?
        .body_mut()
        .read_to_string()?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let tag = json["tag_name"].as_str().ok_or_else(|| anyhow!("no tag in the release"))?;
    let version = tag.trim_start_matches('v').to_string();
    parse(&version).ok_or_else(|| anyhow!("release tag {tag} is not a version"))?;
    Ok(Release { version })
}

/// Update settings, kept in `update-prefs.json` in the data folder.
pub struct Prefs {
    pub check_on_start: bool,
    /// A version the user chose to ignore (without `v`).
    pub ignored: String,
}

impl Prefs {
    pub fn load(dir: &Path) -> Prefs {
        let json: serde_json::Value = std::fs::read_to_string(dir.join(PREFS)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        Prefs {
            check_on_start: json["check_on_start"].as_bool().unwrap_or(true),
            ignored: json["ignored"].as_str().unwrap_or("").to_string(),
        }
    }

    pub fn save(&self, dir: &Path) {
        let json = serde_json::json!({ "check_on_start": self.check_on_start, "ignored": self.ignored });
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(PREFS), json.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_versions() {
        assert!(is_newer("0.7.4", "0.7.3"));
        assert!(is_newer("v1.0.0", "0.9.9"));
        assert!(is_newer("0.10.0", "0.9.0"));
        assert!(!is_newer("0.7.3", "0.7.3"));
        assert!(!is_newer("0.7.2", "0.7.3"));
        assert!(is_newer("0.8.0", "0.8.0-beta.1"));
        assert!(!is_newer("0.8.0-beta.1", "0.8.0"));
        assert!(!is_newer("junk", "0.7.3"));
        assert!(!is_newer("0.7.4", "dev-abc1234"));
    }

    #[test]
    fn prefs_round_trip() {
        let dir = std::env::temp_dir().join(format!("mosaic-update-test-{}", std::process::id()));
        assert!(Prefs::load(&dir).check_on_start);
        Prefs { check_on_start: false, ignored: "0.7.4".into() }.save(&dir);
        let p = Prefs::load(&dir);
        assert!(!p.check_on_start);
        assert_eq!(p.ignored, "0.7.4");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
