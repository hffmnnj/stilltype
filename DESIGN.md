---
version: alpha
name: Stilltype
description: A quiet desktop instrument that turns silently mouthed words into local text.
colors:
  primary: "#D4381C"
  on-primary: "#FFFFFF"
  neutral: "#F4EEE2"
  surface: "#FBF7EE"
  on-surface: "#1B1712"
  muted: "#5E5548"
  error: "#B72D14"
  on-error: "#FFFFFF"
  stage: "#1B1712"
  on-stage: "#EFE7D8"
typography:
  display:
    fontFamily: Newsreader, Georgia, serif
    fontSize: 34px
    fontWeight: 500
    lineHeight: 1.15
  headline:
    fontFamily: system-ui, sans-serif
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: system-ui, sans-serif
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: system-ui, sans-serif
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.4
rounded:
  sm: 6px
  md: 14px
  lg: 20px
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
    rounded: "{rounded.lg}"
    typography: "{typography.display}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.sm}"
    height: 40px
  error:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.error}"
  error-banner:
    backgroundColor: "{colors.error}"
    textColor: "{colors.on-error}"
---

# Stilltype

## Overview

Stilltype is a local-first silent dictation instrument for libraries, shared rooms and quiet work. The main screen answers whether the camera and model are ready, then turns a silently mouthed sentence into editable text. Its closed-mouth seam is the signature, never a microphone or audio waveform. Tauri 2, Rust and React target Windows, Linux and macOS. Use shadcn/Radix primitives and Lucide icons. This file is the initial token source; keep it and CSS aligned. Reference: Raycast for keyboard economy; no visual identity copied.

## Colors

Parchment background, lighter paper content and ink camera stage. Vermilion belongs to identity, recording and the primary action. Small red text uses the darker error color. Borders are #D9CFBD, with stronger control borders #BDB09A. Success #2F6B3F and warning #8A5A00 are status only. Dark overrides: background #16130F, surface #1E1A15, text #EFE7D8, muted #A89E8D, border #342E25, stronger border #4B4336, primary #FF6A45 with ink foreground, stage #0E0C09, error #FF8C70, success #7FC48F and warning #E3B04B. Never invert the camera stage.

## Typography

Newsreader or bundled serif fallback is only for page titles and transcript reading. System UI is the instrument panel. Use 12, 14, 16, 20 and 34 px; no tiny low-contrast labels. Timings and keycaps use ui-monospace. Keep lines below 80 characters.

## Layout

A 224 px labeled left rail and centered work area with 32–40 px margins. The Dictate stage shares edges with its header and transcript area. At 1000 px, compress margins and optional side content. At 680 px, replace the rail with bottom navigation and stack columns. No horizontal page scroll at 390 px.

## Elevation & Depth

Use hairlines and tone steps, not nested floating cards. Content is opaque. Shadows belong on menus, dialogs and the command menu. The dark stage holds actual video and alignment guidance, not gradients.

## Shapes

Controls have 6 px radius, panels 14 px and the stage 20 px. Record is circular. Lists and settings use quiet rules rather than one card per field.

## Components

- Buttons: one primary record/setup action per area, neutral secondary actions and clearly destructive deletion.
- Dialogs: use Radix focus management, titles and descriptions for consent, history editing and search.
- Switches: labeled Radix switches with visible state.
- Tabs: use Radix tabs only for sibling settings groups.
- Camera stage: real preview when available; deliberate face guide and actionable denied/no-device state otherwise.
- History: date-grouped editable rows, raw/cleaned text, favorite, copy and delete. Never seed invented dictations.
- Model setup: real 2026 USR 2.0 model information, explicit download consent, meaningful progress, GPU-first and CPU selection, and real readiness.

## Do's and Don'ts

- Do keep the name and closed-mouth mark visible for recognition.
- Do show real local state and zero history because fabricated data destroys trust.
- Do derive guidance from the actual face detector.
- Don't use purple gradients, fake accuracy percentages, KPI grids or audio waveforms.
- Don't open the camera on launch; request it through a clear action.
- Don't imply hiding preview turns off capture.

## Behaviors

- Loading keeps shell and content geometry with quiet progress.
- First run gives usable camera/setup controls, not a marketing tour.
- Empty history links to Dictate; no results offers clear search.
- Errors stay near the failed action and include retry.
- Browser preview is explicitly labeled. It persists real local edits but never claims native inference or system shortcuts.
- Escape cancels capture or closes dialogs. Command/Control+K opens commands. Command/Control+Enter toggles capture in the window. Native shortcut emits press and release events.
- Recording shows stop, elapsed time and a limit. Frame writes finish before transcription.
- Data stays local. Downloads are intentional; optional Ollama is loopback-only.
- Respect reduced motion. Transitions are 120–200 ms opacity/transform, no ornamental loops. Focus stays visible.
- Dictate stays mounted during navigation so an explicitly enabled camera and native shortcut continue working; the persistent shell exposes camera state. Recording does not pulse.
- The default beam is 5, balancing GPU memory and responsiveness; advanced users may select 1–60. Auto prefers GPU, explicit GPU never silently falls back, and CPU is a genuine alternative with the same 2026 recognition model.
- Mobile explicitly places scrollable main content in grid row 1 and bottom navigation in row 2; do not rely on DOM order for these positions.
- Empty camera stages show corner markers only. Full face, eye and mouth guides appear only with an open camera so they never cross empty-state copy.

## Surfaces

### Dictate

The stage is the high-contrast brand moment, 4:3 with width capped relative to viewport height. Setup and latest transcript stay useful and compact. YuNet 2026may is a local face-tracking export; no fabricated face/quality scores appear without real detector output.

### History, vocabulary, models and settings

Use paper surfaces and precise rows. Export actual history only. Privacy states retention without promising encryption.

## Decisions

| Date | Decision | Why |
| --- | --- | --- |
| 2026-10-31 | Stilltype with a vermilion closed mouth | Distinct from audio dictation and suited to quiet work. |
| 2026-10-31 | Camera and model readiness lead | Experimental inference must show capabilities honestly. |
