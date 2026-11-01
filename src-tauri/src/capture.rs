use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use uuid::Uuid;

const MAX_FILE: u64 = 512 * 1024 * 1024;
const MAX_FRAMES: usize = 3100;
#[derive(Debug, Serialize, Deserialize)]
pub struct Frame {
    pub jpeg: String,
    pub timestamp: f64,
    pub landmarks: Option<Vec<[f64; 2]>>,
}
struct Clip {
    first: Option<f64>,
    last: Option<f64>,
    bytes: u64,
    frames: usize,
    busy: bool,
    cancelled: bool,
}
#[derive(Default)]
pub struct Clips {
    active: HashMap<String, Clip>,
}
fn clip_dir(root: &Path, id: &str) -> Result<PathBuf, String> {
    let parsed = Uuid::parse_str(id).map_err(|_| "Invalid clip ID")?;
    if parsed.to_string() != id {
        return Err("Invalid clip ID".into());
    }
    Ok(root.join(id))
}
impl Clips {
    pub fn start(&mut self, root: &Path) -> Result<String, String> {
        fs::create_dir_all(root).map_err(|e| e.to_string())?;
        let id = Uuid::new_v4().to_string();
        let dir = clip_dir(root, &id)?;
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        if let Err(e) = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(dir.join("frames.jsonl"))
        {
            let _ = fs::remove_dir(&dir);
            return Err(e.to_string());
        }
        self.active.insert(
            id.clone(),
            Clip {
                first: None,
                last: None,
                bytes: 0,
                frames: 0,
                busy: false,
                cancelled: false,
            },
        );
        Ok(id)
    }
    pub fn append(
        &mut self,
        root: &Path,
        id: &str,
        mut frames: Vec<Frame>,
        max_seconds: u32,
    ) -> Result<(), String> {
        let dir = clip_dir(root, id)?;
        let clip = self.active.get_mut(id).ok_or("Unknown clip")?;
        if clip.busy {
            return Err("Clip is being transcribed".into());
        }
        if frames.is_empty() || frames.len() > 256 || clip.frames + frames.len() > MAX_FRAMES {
            return Err("Invalid frame count".into());
        }
        let mut first = clip.first;
        let mut last = clip.last;
        let mut buffer = Vec::new();
        for frame in &mut frames {
            if !frame.timestamp.is_finite()
                || frame.timestamp < 0.0
                || last.is_some_and(|v| frame.timestamp <= v)
            {
                return Err("Invalid frame timestamp".into());
            }
            let start = *first.get_or_insert(frame.timestamp);
            if frame.timestamp - start > f64::from(max_seconds) * 1000.0 {
                return Err("Clip duration exceeded".into());
            }
            last = Some(frame.timestamp);
            if frame.jpeg.len() > 1_500_000 {
                return Err("Frame too large".into());
            }
            let encoded = frame
                .jpeg
                .strip_prefix("data:image/jpeg;base64,")
                .unwrap_or(&frame.jpeg);
            let jpeg = STANDARD
                .decode(encoded)
                .map_err(|_| "Invalid JPEG encoding")?;
            if jpeg.len() < 4 || !jpeg.starts_with(&[0xff, 0xd8]) || !jpeg.ends_with(&[0xff, 0xd9])
            {
                return Err("Invalid JPEG".into());
            }
            if let Some(encoded) = frame.jpeg.strip_prefix("data:image/jpeg;base64,") {
                frame.jpeg = encoded.to_owned();
            }
            if let Some(points) = &frame.landmarks {
                if points.len() != 4
                    || points
                        .iter()
                        .flatten()
                        .any(|n| !n.is_finite() || n.abs() > 10000.0)
                {
                    return Err("Invalid landmarks".into());
                }
            }
            serde_json::to_writer(&mut buffer, frame).map_err(|e| e.to_string())?;
            buffer.push(b'\n');
            if clip.bytes + buffer.len() as u64 > MAX_FILE {
                return Err("Clip size exceeded".into());
            }
        }
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.join("frames.jsonl"))
            .map_err(|e| e.to_string())?;
        f.write_all(&buffer).map_err(|e| e.to_string())?;
        clip.first = first;
        clip.last = last;
        clip.bytes += buffer.len() as u64;
        clip.frames += frames.len();
        Ok(())
    }
    pub fn begin(&mut self, root: &Path, id: &str) -> Result<PathBuf, String> {
        let dir = clip_dir(root, id)?;
        let clip = self.active.get_mut(id).ok_or("Unknown clip")?;
        if clip.busy || clip.frames == 0 {
            return Err("Clip is busy or empty".into());
        }
        clip.busy = true;
        Ok(dir)
    }
    pub fn finish(&mut self, root: &Path, id: &str, remove: bool) -> Result<bool, String> {
        let dir = clip_dir(root, id)?;
        let clip = self.active.get_mut(id).ok_or("Unknown clip")?;
        let cancelled = clip.cancelled;
        if remove || cancelled {
            fs::remove_file(dir.join("frames.jsonl")).map_err(|e| e.to_string())?;
            fs::remove_dir(dir).map_err(|e| e.to_string())?;
            self.active.remove(id);
        } else {
            clip.busy = false;
        }
        Ok(cancelled)
    }
    pub fn cancel(&mut self, root: &Path, id: &str) -> Result<(), String> {
        let clip = self.active.get_mut(id).ok_or("Unknown clip")?;
        if clip.busy {
            clip.cancelled = true;
            return Ok(());
        }
        self.finish(root, id, true).map(|_| ())
    }
}
pub fn cleanup(root: &Path) -> Result<(), String> {
    match fs::read_dir(root) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(|e| e.to_string())?;
                let name = entry.file_name();
                let Some(id) = name.to_str() else { continue };
                if clip_dir(root, id).is_err()
                    || !entry.file_type().map_err(|e| e.to_string())?.is_dir()
                {
                    continue;
                }
                let dir = entry.path();
                match fs::remove_file(dir.join("frames.jsonl")) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.to_string()),
                }
                match fs::remove_dir(dir) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::DirectoryNotEmpty => {}
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unknown_paths_and_bad_frames_without_writing() {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut clips = Clips::default();
        let id = clips.start(&dir).unwrap();
        let frame = || Frame {
            jpeg: "/9j/2Q==".into(),
            timestamp: 1.0,
            landmarks: Some(vec![[1.0, 2.0]; 4]),
        };
        let unknown_rejected = clips.append(&dir, "../evil", vec![frame()], 30).is_err();
        clips.append(&dir, &id, vec![frame()], 30).unwrap();
        let repeat_rejected = clips.append(&dir, &id, vec![frame()], 30).is_err();
        let invalid_landmarks_rejected = clips
            .append(
                &dir,
                &id,
                vec![Frame {
                    jpeg: "/9j/2Q==".into(),
                    timestamp: 2.0,
                    landmarks: Some(vec![[1.0, 2.0]; 3]),
                }],
                30,
            )
            .is_err();
        let lines = fs::read_to_string(dir.join(&id).join("frames.jsonl"))
            .unwrap()
            .lines()
            .count();
        clips.cancel(&dir, &id).unwrap();
        assert!(!dir.join(id).exists());
        fs::remove_dir(dir).unwrap();
        assert!(unknown_rejected);
        assert!(repeat_rejected);
        assert!(invalid_landmarks_rejected);
        assert_eq!(lines, 1);
    }
    #[test]
    fn two_minute_clip_accepted_and_over_limits_rejected() {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut clips = Clips::default();
        let id = clips.start(&dir).unwrap();
        // 3001 frames at 25 fps: 0, 40, ..., 120000 ms. Each batch respects
        // the 256-frame per-append limit and the 120 s duration limit.
        let mut ts = 0.0;
        let mut appended = 0;
        while appended < 3001 {
            let take = (3001 - appended).min(256);
            let batch = (0..take)
                .map(|_| {
                    let f = Frame {
                        jpeg: "/9j/2Q==".into(),
                        timestamp: ts,
                        landmarks: None,
                    };
                    ts += 40.0;
                    f
                })
                .collect::<Vec<_>>();
            clips.append(&dir, &id, batch, 120).unwrap();
            appended += take;
        }
        assert_eq!(appended, 3001);
        // One frame past the 120 s window is rejected by duration...
        assert!(clips
            .append(
                &dir,
                &id,
                vec![Frame {
                    jpeg: "/9j/2Q==".into(),
                    timestamp: 120_040.0,
                    landmarks: None,
                }],
                120
            )
            .is_err());
        // ...and frames past the finite frame cap are rejected even when a
        // generous duration limit would allow them.
        let mut extra = 0;
        let mut capped = false;
        while extra < 200 {
            let batch = (0..50)
                .map(|_| {
                    let f = Frame {
                        jpeg: "/9j/2Q==".into(),
                        timestamp: ts,
                        landmarks: None,
                    };
                    ts += 1.0;
                    f
                })
                .collect::<Vec<_>>();
            if clips.append(&dir, &id, batch, 3600).is_err() {
                capped = true;
                break;
            }
            extra += 50;
        }
        assert!(capped, "frame cap must stay finite above 3001 frames");
        clips.cancel(&dir, &id).unwrap();
        assert!(!dir.join(id).exists());
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn cancellation_waits_for_active_reader_then_removes_clip() {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut clips = Clips::default();
        let id = clips.start(&dir).unwrap();
        clips
            .append(
                &dir,
                &id,
                vec![Frame {
                    jpeg: "/9j/2Q==".into(),
                    timestamp: 0.0,
                    landmarks: None,
                }],
                30,
            )
            .unwrap();
        let clip_dir = clips.begin(&dir, &id).unwrap();
        clips.cancel(&dir, &id).unwrap();
        assert!(clip_dir.join("frames.jsonl").exists());
        assert!(clips.finish(&dir, &id, false).unwrap());
        assert!(!clip_dir.exists());
        fs::remove_dir(dir).unwrap();
    }
}
