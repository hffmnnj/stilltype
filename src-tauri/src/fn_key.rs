use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct FnMonitor {
    active: AtomicBool,
    stop: AtomicBool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShortcutStatus {
    fn_supported: bool,
    fn_active: bool,
    message: String,
}

#[tauri::command]
pub fn shortcut_status(state: tauri::State<'_, std::sync::Arc<FnMonitor>>) -> ShortcutStatus {
    let active = state.active.load(Ordering::Acquire);
    #[cfg(target_os = "macos")]
    let message = if active {
        "Fn monitoring is active."
    } else if !unsafe { macos::CGPreflightListenEventAccess() } {
        "Fn monitoring needs Input Monitoring permission; enable it in macOS settings and restart Stilltype. The configured shortcut remains available."
    } else {
        "Fn monitoring is unavailable. The configured shortcut remains available."
    };
    #[cfg(not(target_os = "macos"))]
    let message =
        "Global Fn monitoring is unsupported on this platform. Use the configured shortcut.";
    ShortcutStatus {
        fn_supported: cfg!(target_os = "macos"),
        fn_active: active,
        message: message.into(),
    }
}

pub fn install(app: &tauri::AppHandle, monitor: std::sync::Arc<FnMonitor>) {
    #[cfg(target_os = "macos")]
    macos::install(app.clone(), monitor);
    #[cfg(not(target_os = "macos"))]
    let _ = (app, monitor);
}

pub fn stop(monitor: &FnMonitor) {
    monitor.stop.store(true, Ordering::Release);
}

#[cfg(target_os = "macos")]
mod macos {
    use super::FnMonitor;
    use std::{
        ffi::c_void,
        ptr,
        sync::{atomic::Ordering, Arc},
    };

    type Ref = *const c_void;
    type Event = *mut c_void;
    type Callback = unsafe extern "C" fn(Ref, u32, Event, *mut c_void) -> Event;
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        pub fn CGPreflightListenEventAccess() -> bool;
        fn CGEventTapCreate(
            tap: u32,
            place: u32,
            options: u32,
            events: u64,
            callback: Callback,
            info: *mut c_void,
        ) -> Ref;
        fn CGEventGetFlags(event: Event) -> u64;
        fn CGEventGetIntegerValueField(event: Event, field: u32) -> i64;
        fn CGEventTapEnable(tap: Ref, enable: bool);
        fn CGEventTapIsEnabled(tap: Ref) -> bool;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFRunLoopDefaultMode: Ref;
        fn CFMachPortCreateRunLoopSource(allocator: Ref, port: Ref, order: isize) -> Ref;
        fn CFMachPortInvalidate(port: Ref);
        fn CFRunLoopGetCurrent() -> Ref;
        fn CFRunLoopAddSource(rl: Ref, source: Ref, mode: Ref);
        fn CFRunLoopRemoveSource(rl: Ref, source: Ref, mode: Ref);
        fn CFRunLoopRunInMode(mode: Ref, seconds: f64, return_after_source: u8) -> i32;
        fn CFRelease(object: Ref);
    }

    struct Context {
        app: tauri::AppHandle,
        monitor: Arc<FnMonitor>,
        tap: Ref,
    }
    unsafe extern "C" fn callback(
        _proxy: Ref,
        kind: u32,
        event: Event,
        info: *mut c_void,
    ) -> Event {
        let context = &*(info as *const Context);
        if kind == 12 && CGEventGetIntegerValueField(event, 9) == 63 {
            let pressed = CGEventGetFlags(event) & (1 << 23) != 0;
            crate::overlay::shortcut(&context.app, pressed, "fn");
        } else if kind == 0xffff_fffe || kind == 0xffff_ffff {
            context.monitor.active.store(false, Ordering::Release);
            crate::overlay::shortcut(&context.app, false, "fn");
            if !context.monitor.stop.load(Ordering::Acquire) {
                CGEventTapEnable(context.tap, true);
                context
                    .monitor
                    .active
                    .store(CGEventTapIsEnabled(context.tap), Ordering::Release);
            }
        }
        event
    }

    pub fn install(app: tauri::AppHandle, monitor: Arc<FnMonitor>) {
        if !unsafe { CGPreflightListenEventAccess() } {
            return;
        }
        std::thread::spawn(move || unsafe {
            let mut context = Box::new(Context {
                app,
                monitor,
                tap: ptr::null(),
            });
            // Listen-only session tap receives modifier changes, not typed text.
            let tap = CGEventTapCreate(
                1,
                0,
                1,
                1 << 12,
                callback,
                (&mut *context as *mut Context).cast(),
            );
            if tap.is_null() {
                return;
            }
            context.tap = tap;
            let source = CFMachPortCreateRunLoopSource(ptr::null(), tap, 0);
            if source.is_null() {
                CFMachPortInvalidate(tap);
                CFRelease(tap);
                return;
            }
            let run_loop = CFRunLoopGetCurrent();
            CFRunLoopAddSource(run_loop, source, kCFRunLoopDefaultMode);
            CGEventTapEnable(tap, true);
            context.monitor.active.store(true, Ordering::Release);
            while !context.monitor.stop.load(Ordering::Acquire) {
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.25, 0);
            }
            context.monitor.active.store(false, Ordering::Release);
            CGEventTapEnable(tap, false);
            CFRunLoopRemoveSource(run_loop, source, kCFRunLoopDefaultMode);
            CFMachPortInvalidate(tap);
            CFRelease(source);
            CFRelease(tap);
        });
    }
}
