use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::Path};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CleanupMode {
    #[default]
    Format,
    Intent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub theme: String,
    pub camera_id: String,
    pub mirrored: bool,
    pub show_preview: bool,
    pub shortcut: String,
    pub dictation_mode: String,
    pub auto_paste: bool,
    pub keep_history: bool,
    pub retention_days: u32,
    pub cleanup_enabled: bool,
    pub cleanup_model: String,
    pub cleanup_mode: CleanupMode,
    pub pause_seconds: f64,
    pub beam_size: u32,
    pub inference_device: String,
    pub max_clip_seconds: u32,
    pub context: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: "system".into(),
            camera_id: String::new(),
            mirrored: true,
            show_preview: true,
            shortcut: "CommandOrControl+Shift+Space".into(),
            dictation_mode: "toggle".into(),
            auto_paste: false,
            keep_history: true,
            retention_days: 30,
            cleanup_enabled: false,
            cleanup_model: "auto".into(),
            cleanup_mode: CleanupMode::Format,
            pause_seconds: 0.7,
            beam_size: 5,
            inference_device: "auto".into(),
            max_clip_seconds: 120,
            context: String::new(),
        }
    }
}
impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        if !["system", "light", "dark"].contains(&self.theme.as_str())
            || !["hold", "toggle"].contains(&self.dictation_mode.as_str())
            || !["auto", "gpu", "cpu"].contains(&self.inference_device.as_str())
            || self.camera_id.len() > 1024
            || self.shortcut.is_empty()
            || self.shortcut.len() > 128
            || !crate::cleanup::valid_model_choice(&self.cleanup_model)
            || self.context.len() > 8192
            || !(1..=60).contains(&self.beam_size)
            || !(5..=120).contains(&self.max_clip_seconds)
            || self.retention_days > 3650
            || !self.pause_seconds.is_finite()
            || !(0.3..=3.0).contains(&self.pause_seconds)
        {
            return Err("Invalid settings".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryEntry {
    pub id: String,
    pub text: String,
    pub raw: String,
    pub created_at: String,
    pub duration_ms: u64,
    pub inference_ms: u64,
    pub favorite: bool,
    pub words: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocabularyWord {
    pub id: String,
    pub word: String,
    pub replacement: String,
    pub created_at: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppState {
    pub settings: Settings,
    pub history: Vec<HistoryEntry>,
    pub vocabulary: Vec<VocabularyWord>,
}
#[derive(Serialize, Deserialize)]
struct Store {
    version: u32,
    #[serde(flatten)]
    data: AppState,
}
pub fn valid_id(id: &str) -> bool {
    Uuid::parse_str(id).is_ok()
}
fn date(s: &str) -> bool {
    DateTime::parse_from_rfc3339(s).is_ok()
}
impl HistoryEntry {
    pub fn validate(&mut self) -> Result<(), String> {
        if !valid_id(&self.id)
            || !date(&self.created_at)
            || self.text.len() > 100_000
            || self.raw.len() > 100_000
            || self.duration_ms > 120_000
        {
            return Err("Invalid history entry".into());
        }
        self.words = self.text.split_whitespace().count();
        Ok(())
    }
}
impl VocabularyWord {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_id(&self.id)
            || !date(&self.created_at)
            || self.word.trim().is_empty()
            || self.word.len() > 256
            || self.replacement.len() > 256
        {
            return Err("Invalid vocabulary word".into());
        }
        Ok(())
    }
}
impl AppState {
    pub fn enforce_retention(&mut self, dir: &Path) -> Result<(), String> {
        let previous_count = self.history.len();
        self.prune();
        if self.history.len() != previous_count {
            self.save(dir)?;
        }
        Ok(())
    }

    pub fn prune(&mut self) {
        if self.settings.retention_days > 0 {
            let cutoff = Utc::now() - Duration::days(self.settings.retention_days.into());
            self.history.retain(|entry| {
                DateTime::parse_from_rfc3339(&entry.created_at)
                    .map(|d| d.with_timezone(&Utc) >= cutoff)
                    .unwrap_or(false)
            });
        }
    }
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("state.json");
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e.to_string()),
        };
        let store: Store =
            serde_json::from_slice(&bytes).map_err(|e| format!("Invalid state file: {e}"))?;
        if store.version != 1 {
            return Err("Unsupported state version".into());
        }
        let mut state = store.data;
        state.settings.validate()?;
        for entry in &mut state.history {
            entry.validate()?;
        }
        for word in &state.vocabulary {
            word.validate()?;
        }
        state.enforce_retention(dir)?;
        Ok(state)
    }
    pub fn save(&mut self, dir: &Path) -> Result<(), String> {
        self.prune();
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
        serde_json::to_writer(
            &mut file,
            &Store {
                version: 1,
                data: self.clone(),
            },
        )
        .map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        file.persist(dir.join("state.json"))
            .map_err(|e| e.error.to_string())?;
        Ok(())
    }
    pub fn export(&self, format: &str) -> Result<String, String> {
        match format {
            "json" => serde_json::to_string_pretty(self).map_err(|e| e.to_string()),
            "txt" => Ok(self
                .history
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")),
            "csv" => {
                let mut csv =
                    String::from("id,createdAt,text,raw,durationMs,inferenceMs,favorite,words\r\n");
                for e in &self.history {
                    csv.push_str(
                        &[
                            e.id.clone(),
                            e.created_at.clone(),
                            e.text.clone(),
                            e.raw.clone(),
                            e.duration_ms.to_string(),
                            e.inference_ms.to_string(),
                            e.favorite.to_string(),
                            e.words.to_string(),
                        ]
                        .iter()
                        .map(|s| csv_cell(s))
                        .collect::<Vec<_>>()
                        .join(","),
                    );
                    csv.push_str("\r\n");
                }
                Ok(csv)
            }
            _ => Err("Unsupported export format".into()),
        }
    }
}
fn csv_cell(value: &str) -> String {
    let dangerous = value
        .trim_start_matches(|c: char| c.is_whitespace() || c.is_control())
        .starts_with(['=', '+', '-', '@']);
    let value = if dangerous {
        format!("'{value}")
    } else {
        value.to_string()
    };
    format!("\"{}\"", value.replace('"', "\"\""))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistence_prunes_expired_and_derives_words() {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut s = AppState::default();
        let mut e = HistoryEntry {
            id: Uuid::new_v4().to_string(),
            text: "two words".into(),
            raw: "".into(),
            created_at: (Utc::now() - Duration::days(31)).to_rfc3339(),
            duration_ms: 1,
            inference_ms: 1,
            favorite: false,
            words: 99,
        };
        e.validate().unwrap();
        assert_eq!(e.words, 2);
        s.history.push(e);
        // A snapshot written before these entries expired must be purged on load,
        // not merely hidden in memory until the next unrelated write.
        fs::write(
            dir.join("state.json"),
            serde_json::to_vec(&Store {
                version: 1,
                data: s,
            })
            .unwrap(),
        )
        .unwrap();
        assert!(AppState::load(&dir).unwrap().history.is_empty());
        let persisted: Store =
            serde_json::from_slice(&fs::read(dir.join("state.json")).unwrap()).unwrap();
        assert!(persisted.data.history.is_empty());
        fs::remove_file(dir.join("state.json")).unwrap();
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn two_minute_session_limits_and_legacy_defaults_still_valid() {
        let mut defaults = Settings::default();
        assert_eq!(defaults.max_clip_seconds, 120);
        defaults.validate().unwrap();
        // Legacy saved settings (e.g. 30 s) must keep loading.
        defaults.max_clip_seconds = 30;
        defaults.validate().unwrap();
        defaults.max_clip_seconds = 5;
        defaults.validate().unwrap();
        defaults.max_clip_seconds = 120;
        defaults.validate().unwrap();
        defaults.max_clip_seconds = 121;
        assert!(defaults.validate().is_err());
        defaults.max_clip_seconds = 4;
        assert!(defaults.validate().is_err());

        let mut entry = HistoryEntry {
            id: Uuid::new_v4().to_string(),
            text: "two minute session".into(),
            raw: "".into(),
            created_at: Utc::now().to_rfc3339(),
            duration_ms: 120_000,
            inference_ms: 1,
            favorite: false,
            words: 0,
        };
        entry.validate().unwrap();
        entry.duration_ms = 120_001;
        assert!(entry.validate().is_err());
    }
    #[test]
    fn legacy_cleanup_settings_default_to_disabled_format_and_validate_model_ids() {
        let settings: Settings = serde_json::from_str("{}").unwrap();
        assert!(!settings.cleanup_enabled);
        assert_eq!(settings.cleanup_mode, CleanupMode::Format);
        assert_eq!(settings.cleanup_model, "auto");
        settings.validate().unwrap();
        let mut invalid = settings;
        invalid.cleanup_model = "unrecognized".into();
        assert!(invalid.validate().is_err());
    }
    #[test]
    fn csv_escapes_formulas_and_quotes() {
        let mut s = AppState::default();
        s.history.push(HistoryEntry {
            id: Uuid::new_v4().to_string(),
            text: " =SUM(1,2) \"x\"".into(),
            raw: "ok".into(),
            created_at: Utc::now().to_rfc3339(),
            duration_ms: 1,
            inference_ms: 2,
            favorite: false,
            words: 2,
        });
        let csv = s.export("csv").unwrap();
        assert!(csv.contains("\"' =SUM(1,2) \"\"x\"\"\""));
        assert!(csv.contains("\"ok\""));
    }
}
