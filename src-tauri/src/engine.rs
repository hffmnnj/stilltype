use crate::state::{Settings, VocabularyWord};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    runtime_installed: bool,
    model_installed: bool,
    ready: bool,
    device: String,
    error: Option<String>,
    data_dir: String,
    uv_available: bool,
}
#[derive(Serialize, Clone)]
struct Progress {
    stage: String,
    message: String,
    progress: f64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultData {
    pub text: String,
    raw: String,
    duration_ms: u64,
    pub inference_ms: u64,
    frames: usize,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_warning: Option<String>,
    #[serde(default, skip_serializing)]
    pub device: Option<String>,
}
pub fn python(data: &Path) -> PathBuf {
    if cfg!(windows) {
        data.join("runtime/Scripts/python.exe")
    } else {
        data.join("runtime/bin/python")
    }
}
pub fn uv() -> Option<PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut dirs = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .unwrap_or_default();
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        dirs.push(PathBuf::from(&home).join(".local/bin"));
        dirs.push(PathBuf::from(home).join(".cargo/bin"));
    }
    dirs.into_iter().map(|p| p.join(name)).find_map(|p| {
        if p.is_file() {
            p.canonicalize().ok()
        } else {
            None
        }
    })
}
pub fn status(data: &Path, worker_alive: bool, device: Option<&str>) -> Status {
    let runtime_installed = python(data).is_file();
    let models = data.join("models");
    let model_installed = fs::read(models.join("installed.json"))
        .ok()
        .filter(|b| b.len() <= 4096)
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|v| {
            v.get("modelId").and_then(Value::as_str) == Some("usr2-baseplus-2026")
                && v.get("releaseYear").and_then(Value::as_u64) == Some(2026)
                && v.get("checkpoint").and_then(Value::as_str) == Some("usr2/baseplus.pth")
                && v.get("sha256").and_then(Value::as_str)
                    == Some("4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773")
                && fs::metadata(models.join("usr2/baseplus.pth"))
                    .is_ok_and(|m| m.is_file() && m.len() == 1_389_096_590)
        });
    Status {
        runtime_installed,
        model_installed,
        ready: runtime_installed && model_installed,
        device: if worker_alive {
            device
                .unwrap_or("Worker running (device not reported)")
                .into()
        } else {
            "Not loaded".into()
        },
        error: None,
        data_dir: data.to_string_lossy().into_owned(),
        uv_available: uv().is_some(),
    }
}
fn emit(app: &AppHandle, stage: &str, message: &str, progress: f64) {
    let _ = app.emit(
        "install-progress",
        Progress {
            stage: stage.into(),
            message: message.into(),
            progress: progress.clamp(0.0, 1.0),
        },
    );
}
fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .stdin(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("Installer command failed: {status}"));
    }
    Ok(())
}
pub fn install(app: &AppHandle, data: &Path, resource: &Path) -> Result<(), String> {
    let result = (|| {
        let uv = uv().ok_or("uv was not found on PATH or in ~/.local/bin or ~/.cargo/bin")?;
        fs::create_dir_all(data).map_err(|e| e.to_string())?;
        emit(app, "runtime", "Creating Python environment", 0.0);
        if !python(data).is_file() {
            run(Command::new(&uv)
                .arg("venv")
                .arg("--allow-existing")
                .arg("--python")
                .arg("3.12")
                .arg(data.join("runtime")))?;
        }
        emit(app, "runtime", "Installing Python dependencies", 0.1);
        let mut pip = Command::new(&uv);
        pip.arg("pip")
            .arg("install")
            .arg("--python")
            .arg(python(data))
            .arg("-r")
            .arg(resource.join("requirements.txt"));
        if !cfg!(target_os = "macos") {
            pip.arg("--torch-backend").arg("auto");
        }
        run(&mut pip)?;
        emit(app, "download", "Downloading models", 0.2);
        let mut child = Command::new(python(data))
            .arg("-u")
            .arg(resource.join("install.py"))
            .arg("--models")
            .arg(data.join("models"))
            .arg("--accept-license")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        if let Some(out) = child.stdout.take() {
            let mut reader = BufReader::new(out);
            loop {
                let mut line = Vec::new();
                let read = (&mut reader)
                    .take(16_385)
                    .read_until(b'\n', &mut line)
                    .map_err(|e| e.to_string())?;
                if read == 0 {
                    break;
                }
                if line.len() > 16_384 {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("Installer progress line too large".into());
                }
                let Ok(line) = String::from_utf8(line) else {
                    continue;
                };
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    let stage = v.get("stage").and_then(Value::as_str).unwrap_or("download");
                    let message = v
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Installing model");
                    let p = v.get("progress").and_then(Value::as_f64).unwrap_or(0.0);
                    if ["runtime", "download", "verify", "ready", "error"].contains(&stage) {
                        emit(
                            app,
                            stage,
                            &message.chars().take(1024).collect::<String>(),
                            p,
                        );
                    }
                }
            }
        }
        let exit = child.wait().map_err(|e| e.to_string())?;
        if !exit.success() {
            return Err(format!("Model installation failed: {exit}"));
        }
        Ok(())
    })();
    if let Err(e) = &result {
        emit(app, "error", e, 0.0)
    }
    result
}
pub struct Worker {
    child: Arc<Mutex<Child>>,
    input: ChildStdin,
    rx: Receiver<Result<String, String>>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Worker {
    fn spawn(
        data: &Path,
        resource: &Path,
        child_slot: &Mutex<Option<Arc<Mutex<Child>>>>,
    ) -> Result<Self, String> {
        let mut child = Command::new(python(data))
            .arg("-u")
            .arg(resource.join("worker.py"))
            .arg("--models")
            .arg(data.join("models"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Could not start worker: {e}"))?;
        let input = child.stdin.take().ok_or("Missing worker input")?;
        let stdout = child.stdout.take().ok_or("Missing worker output")?;
        let (tx, rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut buf = Vec::new();
                let read = (&mut reader)
                    .take(1024 * 1024 + 1)
                    .read_until(b'\n', &mut buf);
                match read {
                    Ok(0) => break,
                    Ok(_) if buf.len() > 1024 * 1024 => {
                        let _ = tx.send(Err("Worker response too large".into()));
                        break;
                    }
                    Ok(_) => {
                        if tx
                            .send(String::from_utf8(buf).map_err(|e| e.to_string()))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e.to_string()));
                        break;
                    }
                }
            }
        });
        let child = Arc::new(Mutex::new(child));
        *child_slot.lock().map_err(|_| "Worker lock failed")? = Some(child.clone());
        Ok(Self { child, input, rx })
    }
    pub fn alive(&mut self) -> bool {
        self.child
            .lock()
            .is_ok_and(|mut child| matches!(child.try_wait(), Ok(None)))
    }
    fn request(&mut self, request: Value, id: &str) -> Result<ResultData, String> {
        serde_json::to_writer(&mut self.input, &request).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())?;
        let line = self
            .rx
            .recv_timeout(Duration::from_secs(900))
            .map_err(|e| format!("Worker stopped or timed out: {e}"))??;
        let response: Value =
            serde_json::from_str(&line).map_err(|e| format!("Invalid worker response: {e}"))?;
        if response.get("id").and_then(Value::as_str) != Some(id) {
            return Err("Worker response ID mismatch".into());
        }
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(response
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Worker failed")
                .to_string());
        }
        serde_json::from_value(
            response
                .get("result")
                .cloned()
                .ok_or("Missing worker result")?,
        )
        .map_err(|e| e.to_string())
    }
}
pub fn transcribe(
    worker: &mut Option<Worker>,
    child_slot: &Mutex<Option<Arc<Mutex<Child>>>>,
    data: &Path,
    resource: &Path,
    clip: &Path,
    final_: bool,
    settings: &Settings,
    vocabulary: &[VocabularyWord],
) -> Result<ResultData, String> {
    if worker.as_mut().is_none_or(|w| !w.alive()) {
        *worker = Some(Worker::spawn(data, resource, child_slot)?);
    }
    let id = Uuid::new_v4().to_string();
    let mut recognition_settings = settings.clone();
    recognition_settings.cleanup_enabled = false;
    let request = json!({"id":id,"command":"transcribe","clipDir":clip,"final":final_,"settings":recognition_settings,"vocabulary":vocabulary});
    let result = worker.as_mut().unwrap().request(request, &id);
    if result.is_err() {
        *worker = None;
    }
    result
}
