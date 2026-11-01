---
version: alpha
name: Stilltype
description: A quiet desktop instrument that turns silently mouthed words into local text.
colors:
  primary: "#171717"
  on-primary: "#FFFFFF"
  neutral: "#FFFFFF"
  surface: "#FAFAFA"
  on-surface: "#171717"
  muted: "#666666"
  error: "#B72D14"
  on-error: "#FFFFFF"
  focus: "#2563EB"
  capture: "#B72D14"
  on-capture: "#FFFFFF"
  stage: "#101010"
  on-stage: "#EDEDED"
typography:
  display:
    fontFamily: Geist Serif, Georgia, serif
    fontSize: 26px
    fontWeight: 400
    lineHeight: 1.25
  headline:
    fontFamily: Geist, system-ui, sans-serif
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: Geist, system-ui, sans-serif
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: Geist, system-ui, sans-serif
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.4
rounded:
  sm: 6px
  md: 12px
  lg: 18px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
components:
  page:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body}"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.md}"
    typography: "{typography.headline}"
  caption:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
  camera-stage:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.on-stage}"
    rounded: "{rounded.md}"
    typography: "{typography.headline}"
  overlay:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.lg}"
    typography: "{typography.body}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.sm}"
    height: 40px
  button-capture:
    backgroundColor: "{colors.capture}"
    textColor: "{colors.on-capture}"
    rounded: "{rounded.sm}"
    height: 40px
  focus-ring:
    backgroundColor: "{colors.focus}"
    textColor: "{colors.neutral}"
  error:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.error}"
  error-banner:
    backgroundColor: "{colors.error}"
    textColor: "{colors.on-error}"
---

# Stilltype

## Overview

Stilltype is a local-first silent dictation instrument for quiet work. The main screen answers whether the camera and model are ready, then turns a silently mouthed sentence into editable text. Its generated brand mark, a warm vermilion mouth on near-black, is the signature, never a microphone or audio waveform. Tauri 2, Rust and React target Windows, Linux and macOS. Use shadcn/Radix primitives and Lucide icons. Keyboard-first: a global shortcut, a command menu and bottom navigation on small screens. This file is the token source; keep it and CSS aligned.

## Colors

White background, quiet gray surface and near-black text. Muted text and hairlines are neutral grays; control borders are a stronger gray. Charcoal is the primary action color with white text. Red is reserved for capture and errors. Focus rings are blue. Success #2F6B3F and warning #8A5A00 are status only. Dark overrides: background #0A0A0A, surface #141414, text #EDEDED, muted #A1A1A1, hairline #262626, stronger border #404040, primary #EDEDED with near-black text, focus #8AB4FF, capture and error #FF8C70, success #7FC48F and warning #E3B04B. Never invert the camera stage.

## Typography

Geist Sans for all controls, body, transcript and navigation, self-hosted with system fallbacks. The standard Geist Serif Regular cut is used sparingly for the wordmark and page/display headings at the restrained 26 px product scale; no large editorial slogan. The overlay uses Geist Sans, never decorative serif. Use 12, 13, 14, 15, 16, 20 and 26 px; no tiny low-contrast labels. Timings and keycaps use ui-monospace. Keep lines below 80 characters.

## Layout

A 200 px labeled left rail and centered work area with 32 px margins. The Dictate stage is compact with a readiness panel beside it. At 1000 px, compress margins and stack the aside. At 680 px, replace the rail with bottom navigation and stack columns. No horizontal page scroll at 390 px.

## Elevation & Depth

Use hairlines and tone steps, not nested floating cards. Content is opaque. Shadows belong on menus, dialogs and the overlay window. The dark stage holds actual video and alignment guidance, not gradients.

## Shapes

Controls have 6 px radius, panels 12 px and the overlay 18 px. Record stays circular. Lists and settings use quiet rules rather than one card per field.

## Components

- Buttons: one primary dictation action per area in charcoal, a red circular record control, neutral secondary actions and clearly destructive deletion.
- Dialogs: use Radix focus management, titles and descriptions for consent, history editing and search.
- Switches: labeled Radix switches with visible state.
- Tabs: use Radix tabs only for sibling settings groups.
- Camera stage: real preview when available; deliberate face guide and actionable denied/no-device state otherwise.
- Overlay: a compact 360 px native window with the generated brand mark, short status text, tabular elapsed time and compact row actions plus quiet dismiss. It never shows a waveform or invented activity.
- History: date-grouped editable rows, raw/cleaned text, favorite, copy and delete. Never seed invented dictations.
- Model setup: real 2026 recognition engine information, explicit download consent, meaningful progress, GPU-first and CPU selection, and real readiness. Transcript cleanup is a bundled local Qwen3.5 text model with its own consent, status and toggle; no network service is involved. Model choice is Auto, 0.8B, 2B or 4B with native size and installed state; Auto is a memory heuristic, not an accuracy ranking, and uses an installed tier before downloading. Format mode fixes casing and punctuation only. Intent mode is experimental, may change wording, falls back to the recognized text with a warning when edits exceed the native guard, and says to review the result. Mode alone never enables cleanup; the toggle needs an installed selected model. The Context field is an optional hint for Intent mode only. Original recognition stays reachable from Dictate and History.
- Preview overlay: the same overlay presentation rendered inline in Settings with live readiness, for inspection without hardware.

## Do's and Don'ts

- Do keep the name and generated brand mark visible for recognition.
- Do show real local state and zero history because fabricated data destroys trust.
- Do derive guidance from the actual face detector.
- Don't use purple gradients, fake accuracy percentages, KPI grids or audio waveforms.
- Don't open the camera on launch; request it through a clear action.
- Don't imply hiding preview turns off capture.
- Don't claim Fn support where firmware reserves the key; report the real shortcut status instead.

## Behaviors

- Loading keeps shell and content geometry with quiet progress.
- First run gives usable camera/setup controls, not a marketing tour.
- Empty history links to Dictate; no results offers clear search.
- Errors stay near the failed action and include retry.
- Browser preview is explicitly labeled. It persists real local edits but never claims native inference, system shortcuts or the overlay window. `?overlay` in a browser renders the same compact overlay with honest readiness and the saved theme, not fake activity; browser actions surface the native-required error, Open returns to the app and Dismiss hides the preview locally.
- Escape cancels capture or closes dialogs in the main window. Command/Control+K opens commands. Command/Control+Enter toggles capture in the window. The native overlay window is non-focusable to preserve the paste target, so it offers no keyboard shortcut; its actions are pointer-driven and the main window keeps its usual Escape behavior. No global Escape shortcut is registered. The native shortcut emits press and release events.
- The overlay follows dictation with one capture owner: main publishes snapshots, the overlay only mirrors. Native shows the window on press and ignores repeat presses; release during startup cancels the pending session via a generation check, and a camera opened solely for that session is released. Dismiss stops visible updates and marks the session canceled in any active startup, recording or processing state, so a late recognition result is dropped instead of saved or pasted. Suppression lasts until a fresh shortcut press or overlay action.
- Fn is always hold-to-dictate; the configured shortcut respects the hold/toggle mode; tray is toggle. On macOS the Fn key needs Input Monitoring permission and a restart after granting it; on Linux and Windows the firmware usually reserves Fn, so the configured shortcut is the reliable trigger.
- A main-window preview action publishes honest current readiness through overlay_action preview without starting capture or opening the camera. Normal in-window capture does not show the overlay.
- Capture runs on a bounded timer cadence, not animation frames, so capture timing is paint-independent; background throttling is still platform-dependent, so hidden-window capture is not guaranteed on every OS.
- Recording shows stop, elapsed time and the configured limit (default 2 minutes, adjustable within 3–120 s on desktop and Android). Frame writes finish before transcription.
- Cleanup status describes the saved model choice: on change, wait for the save, show Checking instead of stale size or readiness, ignore obsolete results, and lock selection and conflicting actions during a download. Consent names the model and size being installed. Browser preview keeps downloads unavailable and readiness false.
- Data stays local. Downloads are intentional. Transcript cleanup runs through the bundled engine only.
- Respect reduced motion. Transitions are 120–180 ms opacity/transform, no ornamental loops. Focus stays visible.
- Dictate stays mounted during navigation so an explicitly enabled camera and native shortcut continue working; the persistent shell exposes camera state. Recording does not pulse.
- The default beam is 5, balancing GPU memory and responsiveness; advanced users may select 1–60. Auto prefers GPU, explicit GPU never silently falls back, and CPU is a genuine alternative with the same 2026 recognition model.
- Readiness never claims ready before the engine status resolves.
- Mobile explicitly places scrollable main content in grid row 1 and bottom navigation in row 2; do not rely on DOM order for these positions.
- Empty camera stages show corner markers only. Full face, eye and mouth guides appear only with an open camera so they never cross empty-state copy.

## Surfaces

### Dictate

A compact readiness and camera panel with one clear main action, a usable transcript area and short contextual guidance. The stage is a restrained 16:10 surface, not an editorial moment. YuNet 2026may is a local face-tracking export; no fabricated face/quality scores appear without real detector output.

### Overlay

A centered charcoal or white control capsule around 360 px wide with generous shadow-safe margins. Status, elapsed time and compact actions share one primary row; a second line appears only for a real result (one line) or error (at most two). With 12 px outer margins and 12 px surface padding the complete surface stays within 112 px height at 420x160. One quiet state-specific action per state (opening has a single Cancel, processing keeps dismiss only) plus a dismiss affordance. Errors wrap without hiding the actions and include a clear Open action to reach the main window. Long result text truncates safely with the full value in an accessible label.

### History, vocabulary, models and settings

Use neutral surfaces and precise rows. Export actual history only. Privacy states retention without promising encryption.

### Native Android

Use Jetpack Compose Material 3 with the same neutral light/dark roles and bundled Geist typography. A compact bottom bar carries Dictate, History, Vocabulary and Settings; model readiness belongs to Dictate and the Models destination reachable from setup and Settings. Camera and model states are real and permission-driven. Quick Settings and a pinned launcher shortcut open a compact capture activity; do not request draw-over-apps or Accessibility privileges. The capture activity returns to the previous app after Copy or Share. No microphone is requested. Stop the camera on backgrounding and cancellation.

Android keeps editable history, favorites, vocabulary, retention, TXT/JSON/CSV export and theme/device preferences. Use 48 dp targets, safe-area insets, flexible text and platform Back. Never seed demo transcripts. Recording sessions run up to the configured limit (default 2 minutes); Android retains up to 3001 detected mouth frames per session while recognition scores them in internal 200-frame model windows over 8-second inference chunks. The 8-second figure is an internal inference window, never a user-facing auto-stop. Mobile recognition has two explicit modes. Fast is the optimized CTC inference mode of the 2026 USR2 weights and stays the default. Quality is experimental: full beam50 decoding of an exported 2026 FLARE Base model under CC-BY-NC-4.0 (non-commercial, not newly trained weights), costing more time and memory. Do not carry desktop beam-search accuracy, H200 speed figures, accuracy percentages or a GPU claim into phone copy. The Models surface shows each pack's own verification and preparation, not a fictional download; Quality preparation needs its own non-commercial terms checkbox and an optional link to the public license in the system browser. Settings offers Fast/Quality under Recording; Quality is selectable only when both packs are installed, because Fast still runs the quiet/activity veto, otherwise it points to model setup. Dictate and the quick sheet show the selected mode and readiness, and Record is disabled until it is ready. A session snapshots its mode at start: changing Settings never alters a running session or relabels a result, and failures are never silently downgraded to Fast. While Quality processes, show real completed/total model windows and an always-available Cancel; Cancel, pause and leaving Dictate drop the session with no late save, copy or display. Result details name the mode, label Quality time as processing time, and state mode-specific provenance. Quality needs about 8 GB of RAM; the backend reports a fixed `unavailableReason` and the UI never computes its own threshold. When present it outranks every setup state: Models shows a neutral "Unavailable" chip with the requirement once (no prepare action, no failure box, terms stay reachable), Settings disables Quality and states the requirement, and Dictate and the quick sheet show "Quality unavailable" with a one-tap "Use Fast" that saves the Fast preference through the normal settings path. A saved Quality preference is never switched silently, and start is refused with the same reason. `quality-memory-low` from a run is a hardware message when `unavailableReason` is set, otherwise a retry-or-use-Fast low-memory message.

## Decisions

| Date | Decision | Why |
| --- | --- | --- |
| 2026-10-31 | Stilltype with a closed-mouth mark | Distinct from audio dictation and suited to quiet work. |
| 2026-10-31 | Camera and model readiness lead | Experimental inference must show capabilities honestly. |
| 2026-11-01 | Neutral system-UI direction, no serif or editorial voice | Requested Vercel, Stripe and Apple quality; monochrome tokens with restrained functional color. Superseded the same day: user asked for Geist Sans and the new Geist Serif (see next row). |
| 2026-11-01 | Geist Sans + Geist Serif Regular | User request; Sans for controls, body, transcript and overlay, Serif Regular for the wordmark and page headings only at 26 px. |
| 2026-11-01 | Native overlay with main-owned snapshots | Follows dictation from any app without invented activity; one capture owner keeps event ordering race-safe. |
| 2026-11-01 | Two-minute recording limit on desktop and Android | User request; 8 seconds remains an internal inference window only, never a user-facing auto-stop. |
| 2026-11-01 | Android Fast/Quality recognition modes | Quality (FLARE Base, beam50, CC-BY-NC-4.0) is opt-in and separately prepared; Fast remains default and the activity veto. Mode is snapshotted per session so results are labelled truthfully. |
| 2026-11-01 | Bundled transcript cleanup, no network service | Local-first cleanup through the bundled engine with explicit consent and status. |
| 2026-11-01 | Generated raster brand mark, no drawn geometry | User request: generated warm vermilion mouth on near-black. Source public/brand/stilltype-icon-generated.png; web app mark, favicon and decorative seam use the public/brand/stilltype-mark.png derivative. Android uses drawable-nodpi/stilltype_brand.png and stilltype_launcher.png with bitmap wrappers; Quick Settings uses the platform camera glyph. No SVG, Canvas or Path brand art remains; camera guides and functional icons are unchanged. |
