use llama_cpp_2::{
    context::params::LlamaContextParams,
    list_llama_ggml_backend_devices,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
    sampling::LlamaSampler,
    LlamaBackendDevice, LlamaBackendDeviceType,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tauri::{AppHandle, Emitter};

pub const MODEL_ID: &str = "qwen3.5-0.8b-2026-q4";
const MAX_GENERATED: usize = 512;
const MAX_INPUT_BYTES: usize = 8192;
const TIME_LIMIT: Duration = Duration::from_secs(45);
const MODEL_SIZE: u64 = 579_615_840;
const MODEL_SHA: &str = "fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526";
const MODEL_URL: &str = "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/f36b1ea49a332ede8fe5f389bbf5b3575ef71f48/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf";

fn model_dir(data: &Path) -> PathBuf {
    data.join("models/cleanup")
}
fn model_path(data: &Path) -> PathBuf {
    model_dir(data).join("qwen3.5-0.8b-q4_k_m.gguf")
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    runtime_bundled: bool,
    model_installed: bool,
    ready: bool,
    model_id: &'static str,
    device: String,
    error: Option<String>,
    total_bytes: u64,
}

#[derive(Clone, Serialize)]
struct Progress {
    stage: &'static str,
    message: String,
    progress: f64,
}

fn emit(app: &AppHandle, stage: &'static str, message: &str, progress: f64) {
    let _ = app.emit(
        "cleanup-install-progress",
        Progress {
            stage,
            message: message.chars().take(512).collect(),
            progress: progress.clamp(0.0, 1.0),
        },
    );
}

pub fn status(data: &Path, cache: &Mutex<Option<Runtime>>) -> Status {
    let marker = model_dir(data).join("installed.sha256");
    let installed = fs::metadata(model_path(data))
        .is_ok_and(|m| m.is_file() && m.len() == MODEL_SIZE)
        && fs::metadata(&marker).is_ok_and(|m| m.is_file() && m.len() == MODEL_SHA.len() as u64)
        && fs::read(marker).is_ok_and(|b| b == MODEL_SHA.as_bytes());
    let device = cache
        .try_lock()
        .ok()
        .and_then(|r| r.as_ref().map(|r| r.device.clone()))
        .unwrap_or_else(|| "Not loaded".into());
    Status {
        runtime_bundled: true,
        model_installed: installed,
        ready: installed,
        model_id: MODEL_ID,
        device,
        error: None,
        total_bytes: MODEL_SIZE,
    }
}

fn verify<R: Read>(
    mut input: R,
    mut output: impl Write,
    expected_size: u64,
    expected_hash: &str,
    mut progress: impl FnMut(u64),
) -> Result<(), String> {
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let limit = (expected_size - total)
            .saturating_add(1)
            .min(buf.len() as u64) as usize;
        let n = input
            .read(&mut buf[..limit])
            .map_err(|e| format!("Could not read cleanup model: {}", e.kind()))?;
        if n == 0 {
            break;
        }
        total = total
            .checked_add(n as u64)
            .ok_or("Cleanup model too large")?;
        if total > expected_size {
            return Err("Cleanup model exceeds expected size".into());
        }
        hasher.update(&buf[..n]);
        output.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        progress(total);
    }
    if total != expected_size {
        return Err("Cleanup model size does not match".into());
    }
    if format!("{:x}", hasher.finalize()) != expected_hash {
        return Err("Cleanup model checksum does not match".into());
    }
    Ok(())
}

pub fn install(app: &AppHandle, data: &Path, cache: &Mutex<Option<Runtime>>) -> Result<(), String> {
    let result: Result<(), String> = (|| {
        let dir = model_dir(data);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        emit(app, "runtime", "Cleanup runtime is bundled", 0.0);
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .https_only(true)
            .redirect(reqwest::redirect::Policy::limited(10))
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(3600))
            .build()
            .map_err(|e| e.to_string())?;
        emit(app, "download", "Downloading cleanup model", 0.0);
        let response = client
            .get(MODEL_URL)
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| format!("Cleanup download failed: {}", e.without_url()))?;
        if response.content_length().is_some_and(|n| n != MODEL_SIZE) {
            return Err("Cleanup download size does not match".into());
        }
        let mut file = tempfile::NamedTempFile::new_in(&dir).map_err(|e| e.to_string())?;
        let mut last_percent = 0;
        verify(response, &mut file, MODEL_SIZE, MODEL_SHA, |n| {
            let percent = (n * 100 / MODEL_SIZE) as u32;
            if percent > last_percent {
                last_percent = percent;
                emit(
                    app,
                    "download",
                    "Downloading cleanup model",
                    f64::from(percent) / 100.0,
                );
            }
        })?;
        emit(app, "verify", "Verified cleanup model checksum", 1.0);
        file.flush()
            .and_then(|_| file.as_file().sync_all())
            .map_err(|e| e.to_string())?;
        let mut marker = tempfile::NamedTempFile::new_in(&dir).map_err(|e| e.to_string())?;
        marker
            .write_all(MODEL_SHA.as_bytes())
            .and_then(|_| marker.as_file().sync_all())
            .map_err(|e| e.to_string())?;
        let mut loaded = cache.lock().map_err(|_| "Cleanup lock failed")?;
        *loaded = None;
        file.persist(model_path(data))
            .map_err(|e| e.error.to_string())?;
        marker
            .persist(dir.join("installed.sha256"))
            .map_err(|e| e.error.to_string())?;
        emit(app, "ready", "Cleanup model installed", 1.0);
        Ok(())
    })();
    if let Err(error) = &result {
        emit(app, "error", error, 0.0);
    }
    result
}

pub fn run(
    cache: Arc<Mutex<Option<Runtime>>>,
    data: PathBuf,
    mode: String,
    text: String,
) -> Result<String, String> {
    let started = Instant::now();
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("stilltype-cleanup".into())
        .spawn(move || {
            let result = (|| {
                let mut runtime = cache.try_lock().map_err(|_| "Cleanup is busy")?;
                if runtime.as_ref().is_some_and(|r| r.mode != mode) {
                    *runtime = None;
                }
                if runtime.is_none() {
                    if !status(&data, &Mutex::new(None)).model_installed {
                        return Err("Install the cleanup model first".into());
                    }
                    let file = fs::File::open(model_path(&data)).map_err(|e| e.to_string())?;
                    if let Err(error) = verify(file, std::io::sink(), MODEL_SIZE, MODEL_SHA, |_| {})
                    {
                        let _ = fs::remove_file(model_dir(&data).join("installed.sha256"));
                        return Err(error);
                    }
                    if started.elapsed() >= TIME_LIMIT {
                        return Err("Cleanup timed out".into());
                    }
                    let path = model_path(&data);
                    *runtime = Some(match Runtime::load(&path, &mode) {
                        Ok(r) => r,
                        Err(_) if mode == "auto" => {
                            let mut r = Runtime::load(&path, "cpu")?;
                            r.mode = mode.clone();
                            r
                        }
                        Err(e) => return Err(e),
                    });
                }
                if started.elapsed() >= TIME_LIMIT {
                    return Err("Cleanup timed out".into());
                }
                runtime.as_ref().unwrap().generate(&text)
            })();
            let _ = tx.send(result);
        })
        .map_err(|e| format!("Could not start cleanup: {e}"))?;
    rx.recv_timeout(TIME_LIMIT).map_err(|error| match error {
        std::sync::mpsc::RecvTimeoutError::Timeout => "Cleanup timed out".to_string(),
        std::sync::mpsc::RecvTimeoutError::Disconnected => {
            "Cleanup stopped unexpectedly".to_string()
        }
    })?
}

pub struct Runtime {
    model: LlamaModel,
    backend: LlamaBackend,
    gpu: bool,
    pub device: String,
    pub mode: String,
}

fn choose_gpu(
    devices: Vec<LlamaBackendDevice>,
    mode: &str,
) -> Result<Option<LlamaBackendDevice>, String> {
    if mode == "cpu" {
        return Ok(None);
    }
    let gpu = devices.into_iter().find(|d| {
        let description = d.description.to_lowercase();
        matches!(
            d.device_type,
            LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
        ) && !["llvmpipe", "lavapipe", "swiftshader"]
            .iter()
            .any(|name| description.contains(name))
    });
    if mode == "gpu" && gpu.is_none() {
        return Err("No supported GPU is available for cleanup".into());
    }
    Ok(gpu)
}

impl Runtime {
    fn load(path: &Path, mode: &str) -> Result<Self, String> {
        let mut backend = LlamaBackend::init().map_err(|e| e.to_string())?;
        backend.void_logs();
        let selected = choose_gpu(list_llama_ggml_backend_devices(), mode)?;
        let gpu = selected.is_some();
        let device = selected.as_ref().map_or_else(
            || "CPU".into(),
            |d| format!("{} ({})", d.description, d.backend),
        );
        let mut params =
            LlamaModelParams::default().with_n_gpu_layers(if selected.is_some() { 999 } else { 0 });
        if let Some(gpu) = selected {
            params = params
                .with_devices(&[gpu.index])
                .map_err(|e| e.to_string())?;
        } else {
            params = params.with_devices(&[]).map_err(|e| e.to_string())?;
        }
        let model = LlamaModel::load_from_file(&backend, path, &params)
            .map_err(|e| format!("Could not load cleanup model: {e}"))?;
        Ok(Self {
            model,
            backend,
            gpu,
            device,
            mode: mode.into(),
        })
    }

    fn generate(&self, text: &str) -> Result<String, String> {
        if text.len() > MAX_INPUT_BYTES || text.trim().is_empty() {
            return Err("Cleanup input is empty or too long".into());
        }
        let started = Instant::now();
        let vocab = self.model.vocab();
        let mut tokens = vocab.tokenize(b"<|im_start|>system\nAdd capitalization and punctuation only. Do not add, remove, or substitute words. Return only the corrected text.<|im_end|>\n<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello, world.<|im_end|>\n<|im_start|>user\n", false, true);
        tokens.extend(vocab.tokenize(text.as_bytes(), false, false));
        tokens.extend(vocab.tokenize(
            b"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            false,
            true,
        ));
        if tokens.len() + MAX_GENERATED > 2048 {
            return Err("Cleanup input exceeds the model context".into());
        }
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_n_batch(2048)
            .with_n_threads(4)
            .with_n_threads_batch(4)
            .with_offload_kqv(self.gpu)
            .with_op_offload(self.gpu);
        let mut ctx = self
            .model
            .new_context(&self.backend, params)
            .map_err(|e| e.to_string())?;
        let mut batch = LlamaBatch::new(2048, 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|e| e.to_string())?;
        ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        let mut sampler = LlamaSampler::greedy();
        let mut output = Vec::new();
        for pos in tokens.len()..tokens.len() + MAX_GENERATED {
            if started.elapsed() > TIME_LIMIT {
                return Err("Cleanup timed out".into());
            }
            let token = sampler.sample(&ctx, -1);
            if vocab.is_eog(token) {
                let result =
                    String::from_utf8(output).map_err(|_| "Cleanup returned invalid UTF-8")?;
                return validate_output(text, &result);
            }
            output.extend(vocab.token_to_piece(token, true, None));
            if output.len() > MAX_INPUT_BYTES * 2 {
                return Err("Cleanup output is too long".into());
            }
            batch.clear();
            batch
                .add(token, pos as i32, &[0], true)
                .map_err(|e| e.to_string())?;
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        }
        Err("Cleanup output exceeded the token limit".into())
    }
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}
fn validate_output(input: &str, output: &str) -> Result<String, String> {
    let trimmed = output.trim();
    if trimmed.is_empty()
        || trimmed.contains('<')
        || trimmed.contains('>')
        || words(input).is_empty()
        || words(input) != words(trimmed)
    {
        return Err("Cleanup output changed the recognized words".into());
    }
    Ok(trimmed.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_must_preserve_words_and_reject_reasoning() {
        assert_eq!(
            validate_output("héllo world", "Héllo, world!").unwrap(),
            "Héllo, world!"
        );
        for output in [
            "hello world",
            "héllo",
            "héllo other world",
            "<think>héllo world</think>",
            "",
        ] {
            assert!(validate_output("héllo world", output).is_err(), "{output}");
        }
    }
    #[test]
    fn download_requires_exact_size_and_sha256() {
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let mut out = Vec::new();
        verify(&b"abc"[..], &mut out, 3, hash, |_| {}).unwrap();
        assert_eq!(out, b"abc");
        assert!(verify(&b"ab"[..], Vec::new(), 3, hash, |_| {}).is_err());
        assert!(verify(&b"abcd"[..], Vec::new(), 3, hash, |_| {}).is_err());
        assert!(verify(&b"xyz"[..], Vec::new(), 3, hash, |_| {}).is_err());
    }
    #[test]
    fn concurrent_cleanup_does_not_queue_another_inference() {
        let cache = Arc::new(Mutex::new(None));
        let _busy = cache.lock().unwrap();
        let result = run(cache.clone(), PathBuf::new(), "cpu".into(), "hello".into());
        assert_eq!(result.unwrap_err(), "Cleanup is busy");
    }
    #[test]
    fn explicit_gpu_does_not_select_cpu_or_software_vulkan() {
        let device = |description: &str, device_type| LlamaBackendDevice {
            index: 0,
            name: "test".into(),
            description: description.into(),
            backend: "Vulkan".into(),
            memory_total: 0,
            memory_free: 0,
            device_type,
        };
        assert!(choose_gpu(
            vec![
                device("CPU", LlamaBackendDeviceType::Cpu),
                device("llvmpipe", LlamaBackendDeviceType::Gpu)
            ],
            "gpu"
        )
        .is_err());
        assert!(choose_gpu(vec![], "auto").unwrap().is_none());
        assert!(
            choose_gpu(vec![device("hardware", LlamaBackendDeviceType::Gpu)], "cpu")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            choose_gpu(vec![device("hardware", LlamaBackendDeviceType::Gpu)], "gpu")
                .unwrap()
                .unwrap()
                .description,
            "hardware"
        );
    }
    #[test]
    #[ignore]
    fn real_model() {
        let path = std::env::var("STILLTYPE_CLEANUP_TEST_MODEL").unwrap();
        let mode = std::env::var("STILLTYPE_CLEANUP_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
        let start = Instant::now();
        verify(
            fs::File::open(&path).unwrap(),
            std::io::sink(),
            MODEL_SIZE,
            MODEL_SHA,
            |_| {},
        )
        .unwrap();
        let verify_elapsed = start.elapsed();
        let runtime = Runtime::load(Path::new(&path), &mode).unwrap();
        let output = runtime
            .generate("hello world this is a local test")
            .unwrap();
        eprintln!(
            "device={} output={output:?} verify={verify_elapsed:?} elapsed={:?}",
            runtime.device,
            start.elapsed()
        );
        assert_ne!(output, "hello world this is a local test");
        assert_eq!(words(&output), words("hello world this is a local test"));
    }
}
