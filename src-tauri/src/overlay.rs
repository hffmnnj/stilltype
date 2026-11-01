use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{Emitter, Manager, PhysicalPosition, State, WebviewWindow};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Ready,
    Opening,
    Recording,
    Processing,
    Done,
    Error,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    Light,
    Dark,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverlaySnapshot {
    pub visible: bool,
    pub phase: Phase,
    pub elapsed: f64,
    pub text: String,
    pub message: String,
    pub camera_open: bool,
    pub theme: Theme,
}

impl Default for OverlaySnapshot {
    fn default() -> Self {
        Self {
            visible: false,
            phase: Phase::Ready,
            elapsed: 0.0,
            text: String::new(),
            message: String::new(),
            camera_open: false,
            theme: Theme::Light,
        }
    }
}

impl OverlaySnapshot {
    fn validate(&self) -> Result<(), String> {
        if !self.elapsed.is_finite()
            || self.elapsed < 0.0
            || self.text.len() > 100_000
            || self.message.len() > 4_000
        {
            return Err("Invalid overlay snapshot".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Inner {
    snapshot: OverlaySnapshot,
    dismissed: bool,
    shortcut_pressed: bool,
    fn_pressed: bool,
}

impl Inner {
    fn shortcut_edge(&mut self, pressed: bool, source: &str) -> bool {
        if source == "tray" {
            return pressed;
        }
        let down = if source == "fn" {
            &mut self.fn_pressed
        } else {
            &mut self.shortcut_pressed
        };
        if *down == pressed {
            return false;
        }
        *down = pressed;
        true
    }
}

#[derive(Default)]
pub struct OverlayState(Mutex<Inner>);

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Preview,
    Start,
    Stop,
    Cancel,
    Open,
    Dismiss,
    Copy,
}

fn show(app: &tauri::AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("overlay")
        .ok_or("Overlay window unavailable")?;
    let monitor = app
        .cursor_position()
        .ok()
        .and_then(|p| app.monitor_from_point(p.x, p.y).ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());
    if let Some(monitor) = monitor {
        let work = monitor.work_area();
        let scale = monitor.scale_factor();
        let width = (420.0 * scale).round() as i32;
        let height = (160.0 * scale).round() as i32;
        let margin = (24.0 * scale).round() as i32;
        let x = work.position.x + ((work.size.width as i32 - width) / 2).max(0);
        let y = work.position.y + (work.size.height as i32 - height - margin).max(0);
        if let Err(e) = window.set_position(PhysicalPosition::new(x, y)) {
            eprintln!("Could not position overlay: {e}");
        }
    }
    show_unfocused(&window)
}

#[cfg(target_os = "linux")]
fn show_unfocused(window: &WebviewWindow) -> Result<(), String> {
    use gtk::prelude::GtkWindowExt;
    window
        .gtk_window()
        .map_err(|e| e.to_string())?
        .set_focus_on_map(false);
    window.show().map_err(|e| e.to_string())
}

#[cfg(windows)]
fn show_unfocused(window: &WebviewWindow) -> Result<(), String> {
    use std::ffi::c_void;
    #[link(name = "user32")]
    extern "system" {
        fn ShowWindow(window: *mut c_void, command: i32) -> i32;
    }
    let handle = window.hwnd().map_err(|e| e.to_string())?;
    unsafe {
        ShowWindow(handle.0, 4);
    }
    Ok(())
}

#[cfg(windows)]
fn hide_window(window: &WebviewWindow) -> Result<(), String> {
    use std::ffi::c_void;
    #[link(name = "user32")]
    extern "system" {
        fn ShowWindow(window: *mut c_void, command: i32) -> i32;
    }
    let handle = window.hwnd().map_err(|e| e.to_string())?;
    unsafe {
        ShowWindow(handle.0, 0);
    }
    Ok(())
}

#[cfg(not(windows))]
fn hide_window(window: &WebviewWindow) -> Result<(), String> {
    window.hide().map_err(|e| e.to_string())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn show_unfocused(window: &WebviewWindow) -> Result<(), String> {
    window.show().map_err(|e| e.to_string())
}

#[cfg(target_os = "macos")]
fn show_unfocused(window: &WebviewWindow) -> Result<(), String> {
    use std::ffi::{c_char, c_void};
    #[link(name = "objc")]
    extern "C" {
        fn sel_registerName(name: *const c_char) -> *const c_void;
        fn objc_msgSend();
    }
    let ns_window = window.ns_window().map_err(|e| e.to_string())?;
    unsafe {
        let selector = sel_registerName(b"orderFront:\0".as_ptr().cast());
        let send: unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) =
            std::mem::transmute(objc_msgSend as *const ());
        send(ns_window, selector, std::ptr::null_mut());
    }
    Ok(())
}

fn refresh_visibility(app: &tauri::AppHandle) -> Result<(), String> {
    let handle = app.clone();
    app.run_on_main_thread(move || {
        let Some(state) = handle.try_state::<OverlayState>() else {
            return;
        };
        let visible = state
            .0
            .lock()
            .map(|inner| inner.snapshot.visible && !inner.dismissed)
            .unwrap_or(false);
        let result = if visible {
            if handle
                .get_webview_window("overlay")
                .is_some_and(|window| window.is_visible().unwrap_or(false))
            {
                Ok(())
            } else {
                show(&handle)
            }
        } else if let Some(window) = handle.get_webview_window("overlay") {
            hide_window(&window)
        } else {
            Err("Overlay window unavailable".into())
        };
        if let Err(e) = result {
            eprintln!("Could not update overlay visibility: {e}");
        }
    })
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_overlay_state(state: State<'_, OverlayState>) -> Result<OverlaySnapshot, String> {
    Ok(state
        .0
        .lock()
        .map_err(|_| "Internal lock failed")?
        .snapshot
        .clone())
}

#[tauri::command]
pub fn update_overlay(
    app: tauri::AppHandle,
    window: WebviewWindow,
    state: State<'_, OverlayState>,
    mut snapshot: OverlaySnapshot,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("Only the main window may publish overlay state".into());
    }
    snapshot.validate()?;
    let mut inner = state.0.lock().map_err(|_| "Internal lock failed")?;
    if inner.dismissed {
        snapshot.visible = false;
    }
    inner.snapshot = snapshot.clone();
    drop(inner);
    app.emit_to("overlay", "overlay-state", &snapshot)
        .map_err(|e| e.to_string())?;
    refresh_visibility(&app)
}

#[tauri::command]
pub fn overlay_action(
    app: tauri::AppHandle,
    window: WebviewWindow,
    state: State<'_, OverlayState>,
    action: Action,
) -> Result<(), String> {
    if window.label() != "overlay" && window.label() != "main" {
        return Err("Invalid window".into());
    }
    match action {
        Action::Dismiss => dismiss(&app, &state)?,
        Action::Preview => {
            let snapshot = {
                let mut inner = state.0.lock().map_err(|_| "Internal lock failed")?;
                inner.dismissed = false;
                inner.snapshot.visible = true;
                inner.snapshot.clone()
            };
            app.emit_to("overlay", "overlay-state", snapshot)
                .map_err(|e| e.to_string())?;
            refresh_visibility(&app)?;
            emit_action(&app, action)?;
        }
        Action::Open => {
            let main = app
                .get_webview_window("main")
                .ok_or("Main window unavailable")?;
            main.show().map_err(|e| e.to_string())?;
            main.set_focus().map_err(|e| e.to_string())?;
        }
        Action::Start => {
            state
                .0
                .lock()
                .map_err(|_| "Internal lock failed")?
                .dismissed = false;
            emit_action(&app, action)?;
        }
        _ => emit_action(&app, action)?,
    }
    Ok(())
}

fn emit_action(app: &tauri::AppHandle, action: Action) -> Result<(), String> {
    app.emit_to(
        "main",
        "overlay-action",
        serde_json::json!({"action": action}),
    )
    .map_err(|e| e.to_string())
}

pub fn dismiss(app: &tauri::AppHandle, state: &OverlayState) -> Result<(), String> {
    let mut inner = state.0.lock().map_err(|_| "Internal lock failed")?;
    inner.dismissed = true;
    inner.snapshot.visible = false;
    drop(inner);
    refresh_visibility(app)?;
    emit_action(app, Action::Dismiss)
}

pub fn shortcut(app: &tauri::AppHandle, pressed: bool, source: &str) {
    let Some(state) = app.try_state::<OverlayState>() else {
        return;
    };
    let Ok(mut inner) = state.0.lock() else {
        return;
    };
    if !inner.shortcut_edge(pressed, source) {
        return;
    }
    if pressed {
        inner.dismissed = false;
        inner.snapshot.visible = true;
    }
    drop(inner);
    if pressed {
        if let Err(e) = refresh_visibility(app) {
            eprintln!("Could not show overlay: {e}");
        }
    }
    let state = if pressed { "pressed" } else { "released" };
    let _ = app.emit_to(
        "main",
        "dictation-shortcut",
        serde_json::json!({"state":state,"source":source}),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_snapshot_bounds_and_non_finite_elapsed() {
        let mut snapshot = OverlaySnapshot::default();
        assert!(snapshot.validate().is_ok());
        snapshot.elapsed = f64::NAN;
        assert!(snapshot.validate().is_err());
        snapshot.elapsed = 1.0;
        snapshot.text = "x".repeat(100_001);
        assert!(snapshot.validate().is_err());
        snapshot.text.clear();
        snapshot.message = "x".repeat(4_001);
        assert!(snapshot.validate().is_err());
    }

    #[test]
    fn repeat_presses_are_suppressed_per_source_but_tray_remains_a_toggle() {
        let mut state = Inner::default();
        assert!(state.shortcut_edge(true, "fn"));
        assert!(!state.shortcut_edge(true, "fn"));
        assert!(state.shortcut_edge(true, "shortcut"));
        assert!(state.shortcut_edge(false, "fn"));
        assert!(!state.shortcut_edge(false, "fn"));
        assert!(state.shortcut_edge(true, "fn"));
        assert!(state.shortcut_edge(true, "tray"));
        assert!(state.shortcut_edge(true, "tray"));
    }
}
