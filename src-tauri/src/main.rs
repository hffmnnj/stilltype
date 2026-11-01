mod capture;
mod engine;
mod state;
use capture::{Clips, Frame};
use enigo::{
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Settings as InputSettings,
};
use serde_json::json;
use state::{AppState, HistoryEntry, Settings, VocabularyWord};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Emitter, Manager, State,
};
use tauri_plugin_clipboard_manager::ClipboardExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

struct Core {
    data: PathBuf,
    resource: PathBuf,
    state: Mutex<AppState>,
    clips: Mutex<Clips>,
    worker: Mutex<Option<engine::Worker>>,
    worker_child: Mutex<Option<Arc<Mutex<std::process::Child>>>>,
    device: Mutex<Option<String>>,
    installing: AtomicBool,
}
fn lock<T>(m: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    m.lock().map_err(|_| "Internal lock failed".into())
}
fn change<T>(core: &Core, f: impl FnOnce(&mut AppState) -> Result<T, String>) -> Result<T, String> {
    let mut s = lock(&core.state)?;
    let mut next = s.clone();
    let out = f(&mut next)?;
    next.save(&core.data)?;
    *s = next;
    Ok(out)
}
#[tauri::command]
fn get_state(core: State<'_, Arc<Core>>) -> Result<AppState, String> {
    let mut state = lock(&core.state)?;
    state.enforce_retention(&core.data)?;
    Ok(state.clone())
}
#[tauri::command]
fn save_settings(
    app: tauri::AppHandle,
    core: State<'_, Arc<Core>>,
    settings: Settings,
) -> Result<Settings, String> {
    settings.validate()?;
    let mut s = lock(&core.state)?;
    let old = s.settings.shortcut.clone();
    let changed = old != settings.shortcut;
    let had_old = app.global_shortcut().is_registered(old.as_str());
    let registered_new = changed || !had_old;
    if registered_new {
        app.global_shortcut()
            .register(settings.shortcut.as_str())
            .map_err(|e| format!("Could not register shortcut: {e}"))?;
        if changed && had_old {
            if let Err(e) = app.global_shortcut().unregister(old.as_str()) {
                let _ = app.global_shortcut().unregister(settings.shortcut.as_str());
                return Err(e.to_string());
            }
        }
    }
    let mut next = s.clone();
    next.settings = settings.clone();
    if let Err(e) = next.save(&core.data) {
        if registered_new {
            let _ = app.global_shortcut().unregister(settings.shortcut.as_str());
            if changed && had_old {
                let _ = app.global_shortcut().register(old.as_str());
            }
        }
        return Err(e);
    }
    *s = next;
    Ok(settings)
}
#[tauri::command]
fn save_entry(core: State<'_, Arc<Core>>, mut entry: HistoryEntry) -> Result<HistoryEntry, String> {
    entry.validate()?;
    change(&core, |s| {
        if let Some(existing) = s.history.iter_mut().find(|e| e.id == entry.id) {
            *existing = entry.clone()
        } else if s.settings.keep_history {
            s.history.insert(0, entry.clone())
        }
        Ok(entry)
    })
}
#[tauri::command]
fn delete_entry(core: State<'_, Arc<Core>>, id: String) -> Result<(), String> {
    if !state::valid_id(&id) {
        return Err("Invalid ID".into());
    }
    change(&core, |s| {
        s.history.retain(|e| e.id != id);
        Ok(())
    })
}
#[tauri::command]
fn clear_history(core: State<'_, Arc<Core>>) -> Result<(), String> {
    change(&core, |s| {
        s.history.clear();
        Ok(())
    })
}
#[tauri::command]
fn save_word(core: State<'_, Arc<Core>>, word: VocabularyWord) -> Result<VocabularyWord, String> {
    word.validate()?;
    change(&core, |s| {
        if let Some(existing) = s.vocabulary.iter_mut().find(|w| w.id == word.id) {
            *existing = word.clone()
        } else {
            s.vocabulary.push(word.clone())
        }
        Ok(word)
    })
}
#[tauri::command]
fn delete_word(core: State<'_, Arc<Core>>, id: String) -> Result<(), String> {
    if !state::valid_id(&id) {
        return Err("Invalid ID".into());
    }
    change(&core, |s| {
        s.vocabulary.retain(|w| w.id != id);
        Ok(())
    })
}
#[tauri::command]
fn export_data(core: State<'_, Arc<Core>>, format: String) -> Result<String, String> {
    let mut s = lock(&core.state)?;
    s.enforce_retention(&core.data)?;
    s.export(&format)
}
#[tauri::command]
fn engine_status(core: State<'_, Arc<Core>>) -> Result<engine::Status, String> {
    let child = lock(&core.worker_child)?;
    let alive = child.as_ref().is_some_and(|child| {
        child
            .lock()
            .is_ok_and(|mut child| matches!(child.try_wait(), Ok(None)))
    });
    let device = lock(&core.device)?;
    Ok(engine::status(&core.data, alive, device.as_deref()))
}
struct InstallGuard(Arc<Core>);
impl Drop for InstallGuard {
    fn drop(&mut self) {
        self.0.installing.store(false, Ordering::Release)
    }
}
#[tauri::command]
async fn install_engine(
    app: tauri::AppHandle,
    core: State<'_, Arc<Core>>,
    accept_license: bool,
) -> Result<engine::Status, String> {
    if !accept_license {
        return Err("Accept the model license before installation".into());
    }
    if core.installing.swap(true, Ordering::AcqRel) {
        return Err("Installation already in progress".into());
    }
    let core = core.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = InstallGuard(core.clone());
        *lock(&core.worker)? = None;
        *lock(&core.device)? = None;
        engine::install(&app, &core.data, &core.resource)?;
        Ok(engine::status(&core.data, false, None))
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn start_clip(core: State<'_, Arc<Core>>) -> Result<String, String> {
    lock(&core.clips)?.start(&core.data.join("clips"))
}
#[tauri::command]
fn append_frames(
    core: State<'_, Arc<Core>>,
    clip_id: String,
    frames: Vec<Frame>,
) -> Result<(), String> {
    let max = lock(&core.state)?.settings.max_clip_seconds;
    lock(&core.clips)?.append(&core.data.join("clips"), &clip_id, frames, max)
}
#[tauri::command]
fn cancel_clip(core: State<'_, Arc<Core>>, clip_id: String) -> Result<(), String> {
    lock(&core.clips)?.cancel(&core.data.join("clips"), &clip_id)
}
#[tauri::command]
async fn transcribe(
    core: State<'_, Arc<Core>>,
    clip_id: String,
    r#final: bool,
) -> Result<engine::ResultData, String> {
    let core = core.inner().clone();
    let dir = lock(&core.clips)?.begin(&core.data.join("clips"), &clip_id)?;
    let (settings, vocabulary) = {
        let state = lock(&core.state)?;
        (state.settings.clone(), state.vocabulary.clone())
    };
    tauri::async_runtime::spawn_blocking(move || {
        let result = lock(&core.worker).and_then(|mut w| {
            if core.installing.load(Ordering::Acquire) {
                return Err("Engine installation is in progress".into());
            }
            engine::transcribe(
                &mut w,
                &core.worker_child,
                &core.data,
                &core.resource,
                &dir,
                r#final,
                &settings,
                &vocabulary,
            )
        });
        if let Ok(mut device) = core.device.lock() {
            *device = result
                .as_ref()
                .ok()
                .and_then(|r| r.device.as_ref())
                .filter(|d| !d.is_empty() && d.len() <= 128)
                .cloned();
        }
        let remove = r#final || result.is_err();
        let cleanup = lock(&core.clips)
            .and_then(|mut c| c.finish(&core.data.join("clips"), &clip_id, remove));
        match (result, cleanup) {
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(e),
            (Ok(_), Ok(true)) => Err("Clip was cancelled".into()),
            (Ok(result), Ok(false)) => Ok(result),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn copy_text(app: tauri::AppHandle, text: String) -> Result<(), String> {
    if text.len() > 100_000 {
        return Err("Text is too long".into());
    }
    app.clipboard().write_text(text).map_err(|e| e.to_string())
}
#[tauri::command]
async fn paste_text(app: tauri::AppHandle, text: String) -> Result<(), String> {
    if text.len() > 100_000 {
        return Err("Text is too long".into());
    }
    app.clipboard()
        .write_text(text)
        .map_err(|e| e.to_string())?;
    if let Some(window) = app.get_webview_window("main") {
        window.hide().map_err(|e| e.to_string())?;
    }
    tauri::async_runtime::spawn_blocking(|| {
        std::thread::sleep(Duration::from_millis(200));
        let mut input = Enigo::new(&InputSettings::default()).map_err(|e| e.to_string())?;
        let modifier = if cfg!(target_os = "macos") {
            Key::Meta
        } else {
            Key::Control
        };
        input.key(modifier, Press).map_err(|e| e.to_string())?;
        let result = input
            .key(Key::Unicode('v'), Click)
            .map_err(|e| e.to_string());
        let released = input.key(modifier, Release).map_err(|e| e.to_string());
        result.and(released)
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn ollama_models() -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get("http://127.0.0.1:11434/api/tags")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let value: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    Ok(value
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or("Invalid Ollama response")?
        .iter()
        .filter_map(|v| v.get("name").and_then(|v| v.as_str()).map(str::to_owned))
        .collect())
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _, event| {
                    let state = if event.state == ShortcutState::Pressed {
                        "pressed"
                    } else {
                        "released"
                    };
                    let _ = app.emit("dictation-shortcut", json!({"state":state}));
                })
                .build(),
        )
        .on_permission_request(|webview, kind| {
            use tauri::webview::{PermissionKind, PermissionResponse};
            if webview.label() == "main" && matches!(kind, PermissionKind::Camera) {
                PermissionResponse::Allow
            } else {
                PermissionResponse::Deny
            }
        })
        .setup(|app| {
            let data = app.path().app_data_dir()?;
            let resource = app.path().resource_dir()?.join("engine");
            std::fs::create_dir_all(&data)?;
            capture::cleanup(&data.join("clips")).map_err(std::io::Error::other)?;
            let state = AppState::load(&data).map_err(std::io::Error::other)?;
            if let Err(e) = app
                .global_shortcut()
                .register(state.settings.shortcut.as_str())
            {
                eprintln!("Global shortcut unavailable: {e}");
            }
            app.manage(Arc::new(Core {
                data,
                resource,
                state: Mutex::new(state),
                clips: Mutex::new(Clips::default()),
                worker: Mutex::new(None),
                worker_child: Mutex::new(None),
                device: Mutex::new(None),
                installing: AtomicBool::new(false),
            }));
            let show = MenuItem::with_id(app, "show", "Show Stilltype", true, None::<&str>)?;
            let dictate = MenuItem::with_id(app, "dictate", "Dictate", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &dictate, &quit])?;
            let mut tray =
                TrayIconBuilder::new()
                    .menu(&menu)
                    .on_menu_event(|app, event| match event.id.as_ref() {
                        "show" => {
                            if let Some(w) = app.get_webview_window("main") {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                        "dictate" => {
                            let _ = app.emit("dictation-shortcut", json!({"state":"pressed"}));
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone())
            }
            tray.build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            save_settings,
            save_entry,
            delete_entry,
            clear_history,
            save_word,
            delete_word,
            engine_status,
            install_engine,
            start_clip,
            append_frames,
            transcribe,
            cancel_clip,
            copy_text,
            paste_text,
            ollama_models,
            export_data
        ])
        .build(tauri::generate_context!())
        .expect("Failed to build Stilltype")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                if let Some(core) = app.try_state::<Arc<Core>>() {
                    if let Ok(child) = core.worker_child.lock() {
                        if let Some(child) = child.as_ref() {
                            if let Ok(mut child) = child.lock() {
                                let _ = child.kill();
                                let _ = child.wait();
                            }
                        }
                    }
                }
            }
        });
}
