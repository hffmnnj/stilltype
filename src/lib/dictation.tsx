import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from 'react';
import { toast } from 'sonner';
import { api, errorMessage, isDesktop, listenNative } from './api';
import { StilltypeCamera } from './camera';
import { useStore } from './store';
import type {
  CaptureQuality,
  DictationShortcut,
  HistoryEntry,
  OverlayAction,
  OverlayPhase,
  OverlaySnapshot,
  TranscriptionResult,
} from './types';
import { countWords, isTypingTarget, newId } from './utils';

export type Phase = 'idle' | 'opening' | 'ready' | 'recording' | 'processing';

export interface Draft {
  text: string;
  raw: string;
  entryId: string | null;
  durationMs: number;
  inferenceMs: number;
  dirty: boolean;
}

interface DictationValue {
  videoRef: RefObject<HTMLVideoElement | null>;
  phase: Phase;
  cameraOpen: boolean;
  devices: MediaDeviceInfo[];
  refreshDevices: () => Promise<void>;
  quality: CaptureQuality | null;
  error: string | null;
  clearError: () => void;
  elapsed: number;
  draft: Draft | null;
  editDraft: (text: string) => void;
  saveDraft: () => Promise<void>;
  discardDraft: () => void;
  copyDraft: () => Promise<void>;
  pasteDraft: () => Promise<void>;
  blockedReason: string | null;
  openCamera: () => Promise<boolean>;
  closeCamera: () => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  cancel: () => Promise<void>;
  toggle: () => Promise<void>;
  /** Live snapshot for the overlay window and the inline Settings preview. */
  overlaySnapshot: () => OverlaySnapshot;
  /** Desktop: ask native to show the overlay with current readiness (no capture). */
  previewOverlay: () => void;
  /** Hide the overlay and cancel any pending opening/recording session. */
  dismissOverlay: () => Promise<void>;
}

const Ctx = createContext<DictationValue | null>(null);

export function useDictation(): DictationValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useDictation outside provider');
  return v;
}

function cameraMessage(e: unknown): string {
  const name = e instanceof DOMException ? e.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return 'Camera access was blocked. Allow it in your system or browser privacy settings, then try again.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError')
    return 'No matching camera was found. Plug one in or pick another in Settings.';
  if (name === 'NotReadableError') return 'The camera is busy in another app. Close it there and try again.';
  return errorMessage(e);
}

const EMPTY_MESSAGE = 'No words were recognised. Face the camera, mouth each word clearly, and try again.';

function overlayPhaseFor(phase: Phase, error: string | null, draft: Draft | null): OverlayPhase {
  if (error) return 'error';
  if (phase === 'opening') return 'opening';
  if (phase === 'recording') return 'recording';
  if (phase === 'processing') return 'processing';
  return draft ? 'done' : 'ready';
}

export function DictationProvider({ children }: { children: ReactNode }) {
  const { settings, engine, history, saveEntry, setCommandOpen, go, resolvedTheme } = useStore();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const cameraRef = useRef<StilltypeCamera | null>(null);
  const [phase, setPhaseState] = useState<Phase>('idle');
  const phaseRef = useRef<Phase>('idle');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [quality, setQuality] = useState<CaptureQuality | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const engineReadyRef = useRef(false);
  engineReadyRef.current = Boolean(engine?.ready);
  const historyRef = useRef(history);
  historyRef.current = history;

  const clipRef = useRef<string | null>(null);
  const startedAtRef = useRef(0);
  const segmentRef = useRef<Promise<void> | null>(null);
  const stopRequestedRef = useRef(false);
  const cancelledRef = useRef(false);
  const sessionTextRef = useRef(false);
  const appendErrorRef = useRef<string | null>(null);
  // Session intent: every new session takes a generation. Releasing a hold
  // shortcut (or dismissing) while camera/model startup is pending bumps the
  // generation, so the late continuation can never start a recording.
  const sessionGenRef = useRef(0);
  // Generation of the session currently in startup, or null once it records.
  const pendingRef = useRef<number | null>(null);
  // True when this session opened the camera; a cancelled startup closes it.
  const sessionOwnedCameraRef = useRef(false);
  // True while the main window owns a visible native overlay session.
  const overlayOnRef = useRef(false);

  const setPhase = useCallback((p: Phase) => {
    phaseRef.current = p;
    setPhaseState(p);
  }, []);

  const blockedReason = !isDesktop
    ? 'Browser preview can show your camera but cannot transcribe. Use the desktop app to dictate.'
    : !engine?.ready
      ? 'Install the research engine in Models before dictating.'
      : null;

  const refreshDevices = useCallback(async () => {
    try {
      setDevices(await StilltypeCamera.getDevices());
    } catch {
      /* listing is best-effort */
    }
  }, []);

  const getCamera = useCallback((): StilltypeCamera | null => {
    if (cameraRef.current) return cameraRef.current;
    const video = videoRef.current;
    if (!video) return null;
    cameraRef.current = new StilltypeCamera(video, {
      onQuality: setQuality,
      onError: (m) => setError(m),
    });
    return cameraRef.current;
  }, []);

  const openCamera = useCallback(async (gen?: number): Promise<boolean> => {
    if (cameraRef.current?.isOpen) return true;
    const cam = getCamera();
    if (!cam) {
      setError('The camera view is not ready yet. Switch to Dictate and try again.');
      return false;
    }
    setError(null);
    setPhase('opening');
    try {
      const devices = await cam.open(settingsRef.current.cameraId || undefined);
      if (gen !== undefined && gen !== sessionGenRef.current) {
        // Cancelled or superseded while opening: release what we just opened.
        cam.close();
        setQuality(null);
        setPhase('idle');
        return false;
      }
      setDevices(devices);
      setPhase('ready');
      return true;
    } catch (e) {
      if (gen !== undefined && gen !== sessionGenRef.current) {
        // Cancelled while the open was in flight; stay quiet.
        setPhase(cameraRef.current?.isOpen ? 'ready' : 'idle');
        return false;
      }
      setError(cameraMessage(e));
      setPhase('idle');
      return false;
    }
  }, [getCamera, setPhase]);

  const closeCamera = useCallback(() => {
    if (phaseRef.current === 'recording' || phaseRef.current === 'processing') return;
    cameraRef.current?.close();
    setQuality(null);
    setPhase('idle');
  }, [setPhase]);

  // Re-open when the chosen camera changes while the preview is live.
  const lastCameraId = useRef(settings.cameraId);
  useEffect(() => {
    if (lastCameraId.current === settings.cameraId) return;
    lastCameraId.current = settings.cameraId;
    const cam = cameraRef.current;
    if (cam?.isOpen && phaseRef.current === 'ready') {
      cam
        .open(settings.cameraId || undefined)
        .then(setDevices)
        .catch((e) => setError(cameraMessage(e)));
    }
  }, [settings.cameraId]);

  const accept = useCallback(
    async (r: TranscriptionResult, quiet: boolean) => {
      const text = r.text.trim();
      // A dismissed session stays quiet: its late result is never saved,
      // shown, or pasted. Normal captures are unaffected.
      if (cancelledRef.current) return;
      if (r.error) {
        if (!quiet) setError(r.error);
        return;
      }
      if (!text) {
        if (!quiet) setError(EMPTY_MESSAGE);
        return;
      }
      sessionTextRef.current = true;
      const entry: HistoryEntry = {
        id: newId(),
        text,
        raw: r.raw,
        createdAt: new Date().toISOString(),
        durationMs: r.durationMs,
        inferenceMs: r.inferenceMs,
        favorite: false,
        words: countWords(text),
      };
      let entryId: string | null = null;
      if (settingsRef.current.keepHistory) {
        try {
          await saveEntry(entry);
          entryId = entry.id;
        } catch (e) {
          toast.error('Transcript kept on screen but not saved to history', { description: errorMessage(e) });
        }
      }
      setDraft({ text, raw: r.raw, entryId, durationMs: r.durationMs, inferenceMs: r.inferenceMs, dirty: false });
      if (r.cleanupWarning) toast('Cleanup skipped', { description: r.cleanupWarning });
      if (settingsRef.current.autoPaste && isDesktop) {
        try {
          await api.pasteText(text);
        } catch (e) {
          toast.error('Could not paste', { description: errorMessage(e) });
        }
      }
    },
    [saveEntry],
  );

  // Invalidate a session still in startup (opening, or the gap before the
  // first clip starts). Releases a camera this session opened, but never
  // touches a camera that was already open.
  const abortPending = useCallback(() => {
    if (pendingRef.current === null) return;
    pendingRef.current = null;
    sessionGenRef.current += 1;
    cancelledRef.current = true;
    stopRequestedRef.current = true;
    if (sessionOwnedCameraRef.current) {
      sessionOwnedCameraRef.current = false;
      cameraRef.current?.close();
      setQuality(null);
      setPhase('idle');
    } else if (phaseRef.current === 'opening') {
      setPhase(cameraRef.current?.isOpen ? 'ready' : 'idle');
    }
  }, [setPhase]);

  // Latest live values for overlay snapshots, readable from async callbacks.
  const liveRef = useRef({
    phase: 'idle' as Phase,
    error: null as string | null,
    draft: null as Draft | null,
    elapsed: 0,
    cameraOpen: false,
    blocked: null as string | null,
    theme: 'light' as 'light' | 'dark',
  });
  liveRef.current = {
    phase,
    error,
    draft,
    elapsed,
    cameraOpen: phase !== 'idle' && phase !== 'opening',
    blocked: blockedReason,
    theme: resolvedTheme,
  };

  const buildSnapshot = useCallback((visible: boolean): OverlaySnapshot => {
    const live = liveRef.current;
    if (!visible) {
      return { visible: false, phase: 'ready', elapsed: 0, text: '', message: '', cameraOpen: false, theme: live.theme };
    }
    const basePhase = overlayPhaseFor(live.phase, live.error, live.draft);
    // A blocked-but-otherwise-ready overlay reports error so it uses the
    // existing error presentation instead of a Start button.
    const overlayPhase: OverlayPhase = basePhase === 'ready' && live.blocked ? 'error' : basePhase;
    const message =
      live.error ??
      (overlayPhase === 'ready'
        ? 'Ready'
        : overlayPhase === 'error'
          ? (live.blocked ?? '')
          : overlayPhase === 'opening'
            ? 'Opening camera…'
            : overlayPhase === 'processing'
              ? 'Transcribing…'
              : '');
    return {
      visible: true,
      phase: overlayPhase,
      elapsed: live.elapsed,
      text: live.draft?.text ?? '',
      message,
      cameraOpen: live.cameraOpen,
      theme: live.theme,
    };
  }, []);

  const publishSnapshot = useCallback(() => {
    if (!overlayOnRef.current) return;
    void api.updateOverlay(buildSnapshot(true)).catch(() => undefined);
  }, [buildSnapshot]);

  // Keep an owned native overlay on live state: real phase, elapsed, result
  // or error text, and the resolved theme.
  useEffect(() => {
    publishSnapshot();
  }, [phase, error, draft, elapsed, resolvedTheme, blockedReason, publishSnapshot]);

  const previewOverlay = useCallback(() => {
    overlayOnRef.current = true;
    // Native shows the cached snapshot without starting capture or opening
    // the camera, then emits overlay-action preview so this publishes live
    // readiness. Never called from the internal start handler.
    if (isDesktop) void api.overlayAction('preview').catch(() => undefined);
    publishSnapshot();
  }, [publishSnapshot]);

  const dismissOverlay = useCallback(async () => {
    // Stop publishing visible snapshots immediately.
    overlayOnRef.current = false;
    abortPending();
    // Mark the session canceled in any active state so a pending
    // recognition result is dropped by accept() instead of saving or
    // auto-pasting after dismissal.
    if (pendingRef.current !== null || phaseRef.current === 'recording' || phaseRef.current === 'processing') {
      cancelledRef.current = true;
      stopRequestedRef.current = true;
    }
    if (phaseRef.current === 'recording') {
      setPhase('processing');
      try {
        await segmentRef.current;
        const id = clipRef.current;
        clipRef.current = null;
        await cameraRef.current?.stop().catch(() => undefined);
        if (id) await api.cancelClip(id);
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setPhase(cameraRef.current?.isOpen ? 'ready' : 'idle');
      }
    }
    try {
      await api.updateOverlay(buildSnapshot(false));
    } catch {
      // Native may already be gone; the session is stopped regardless.
    }
  }, [abortPending, buildSnapshot, setPhase]);

  const beginClip = useCallback(async (onPause: (() => void) | undefined): Promise<void> => {
    const cam = cameraRef.current;
    if (!cam) throw new Error('Camera is not open.');
    const id = await api.startClip();
    clipRef.current = id;
    cam.start(
      async (frames) => {
        try {
          await api.appendFrames(id, frames);
        } catch (e) {
          appendErrorRef.current = errorMessage(e);
          throw e;
        }
      },
      onPause,
      settingsRef.current.pauseSeconds,
    );
  }, []);

  const finishClip = useCallback(
    async (quiet: boolean) => {
      const id = clipRef.current;
      const cam = cameraRef.current;
      if (!id || !cam) return;
      await cam.stop();
      clipRef.current = null;
      if (appendErrorRef.current) {
        if (!quiet) setError(appendErrorRef.current);
        appendErrorRef.current = null;
        await api.cancelClip(id).catch(() => undefined);
        return;
      }
      const r = await api.transcribe(id, true);
      await accept(r, quiet);
    },
    [accept],
  );

  const runSegment = useCallback(() => {
    if (segmentRef.current) return;
    const p = (async () => {
      try {
        await finishClip(true);
        if (!stopRequestedRef.current && !cancelledRef.current) await beginClip(pauseHandlerRef.current);
      } catch (e) {
        setError(errorMessage(e));
        stopRequestedRef.current = true;
      } finally {
        segmentRef.current = null;
      }
    })();
    segmentRef.current = p;
  }, [beginClip, finishClip]);

  const pauseHandlerRef = useRef<(() => void) | undefined>(undefined);
  pauseHandlerRef.current = () => {
    if (phaseRef.current === 'recording' && !segmentRef.current && !stopRequestedRef.current) runSegment();
  };

  const beginSession = useCallback(async (opts: { hold: boolean; overlay: boolean }) => {
    // Duplicate presses (key repeat, tray, overlay Start) never restart capture.
    if (phaseRef.current === 'recording' || phaseRef.current === 'processing' || phaseRef.current === 'opening') return;
    const gen = sessionGenRef.current + 1;
    sessionGenRef.current = gen;
    pendingRef.current = gen;
    setError(null);
    if (!isDesktop || !engineReadyRef.current) {
      pendingRef.current = null;
      setError(
        !isDesktop
          ? 'Browser preview cannot transcribe. Use the desktop app to dictate.'
          : 'Install the research engine in Models before dictating.',
      );
      // A repeated block sets the identical error, so no state effect fires.
      publishSnapshot();
      return;
    }
    if (opts.overlay) overlayOnRef.current = true;
    sessionOwnedCameraRef.current = !cameraRef.current?.isOpen;
    cancelledRef.current = false;
    stopRequestedRef.current = false;
    sessionTextRef.current = false;
    appendErrorRef.current = null;
    if (!(await openCamera(gen))) {
      pendingRef.current = null;
      sessionOwnedCameraRef.current = false;
      return;
    }
    // Released or dismissed while startup was pending: stay stopped and
    // release a camera opened solely for this session.
    if (gen !== sessionGenRef.current || cancelledRef.current) {
      pendingRef.current = null;
      if (sessionOwnedCameraRef.current) {
        sessionOwnedCameraRef.current = false;
        cameraRef.current?.close();
        setQuality(null);
        setPhase('idle');
      }
      return;
    }
    // Ownership stays set across the clip-start gap so a release in that
    // window still closes a session-owned camera. Cleared once recording
    // actually begins, or on the paths below.
    try {
      // Fn is always hold-to-dictate, so it never segments; toggle mode splits.
      await beginClip(opts.hold ? undefined : () => pauseHandlerRef.current?.());
      if (gen !== sessionGenRef.current || cancelledRef.current) {
        // Released or dismissed while the clip was starting: cancel the
        // newly allocated clip and release a session-owned camera, while a
        // camera that was already open stays open.
        const id = clipRef.current;
        clipRef.current = null;
        await cameraRef.current?.stop().catch(() => undefined);
        if (id) await api.cancelClip(id).catch(() => undefined);
        pendingRef.current = null;
        if (sessionOwnedCameraRef.current) {
          sessionOwnedCameraRef.current = false;
          cameraRef.current?.close();
          setQuality(null);
        }
        setPhase(cameraRef.current?.isOpen ? 'ready' : 'idle');
        return;
      }
      sessionOwnedCameraRef.current = false;
      pendingRef.current = null;
      startedAtRef.current = performance.now();
      setElapsed(0);
      setPhase('recording');
    } catch (e) {
      pendingRef.current = null;
      sessionOwnedCameraRef.current = false;
      setError(errorMessage(e));
      setPhase(cameraRef.current?.isOpen ? 'ready' : 'idle');
    }
  }, [beginClip, openCamera, setPhase, publishSnapshot]);

  const start = useCallback(async () => {
    // In-window capture stays local: it never asks native to show the overlay.
    await beginSession({ hold: settingsRef.current.dictationMode === 'hold', overlay: false });
  }, [beginSession]);

  const stop = useCallback(async () => {
    if (phaseRef.current !== 'recording') return;
    stopRequestedRef.current = true;
    setPhase('processing');
    try {
      await segmentRef.current;
      const quiet = sessionTextRef.current; // earlier segments already produced text
      await finishClip(quiet);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPhase('ready');
    }
  }, [finishClip, setPhase]);

  const cancel = useCallback(async () => {
    if (pendingRef.current !== null || phaseRef.current === 'opening') {
      abortPending();
      toast('Capture cancelled');
      return;
    }
    if (phaseRef.current !== 'recording') return;
    cancelledRef.current = true;
    stopRequestedRef.current = true;
    setPhase('processing');
    try {
      await segmentRef.current;
      const id = clipRef.current;
      clipRef.current = null;
      await cameraRef.current?.stop().catch(() => undefined);
      if (id) await api.cancelClip(id);
      toast('Capture cancelled');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPhase('ready');
    }
  }, [setPhase]);

  const toggle = useCallback(async () => {
    if (phaseRef.current === 'recording') await stop();
    else await start();
  }, [start, stop]);

  // Enforce the clip limit.
  useEffect(() => {
    if (phase !== 'recording') return;
    const t = window.setInterval(() => {
      const s = (performance.now() - startedAtRef.current) / 1000;
      setElapsed(s);
      if (s >= settingsRef.current.maxClipSeconds) void stop();
    }, 250);
    return () => window.clearInterval(t);
  }, [phase, stop]);

  // Native global shortcut (main window only).
  // Explicit publish: native shows its cached snapshot at once, and a
  // repeated block may leave state unchanged.
  const publishRef = useRef(publishSnapshot);
  publishRef.current = publishSnapshot;
  const toggleRef = useRef(toggle);
  toggleRef.current = toggle;
  const beginSessionRef = useRef(beginSession);
  beginSessionRef.current = beginSession;
  const abortPendingRef = useRef(abortPending);
  abortPendingRef.current = abortPending;
  const stopRef = useRef(stop);
  stopRef.current = stop;
  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    void listenNative<DictationShortcut>('dictation-shortcut', ({ state, source }) => {
      // Fn is always hold-to-dictate; the configured shortcut respects the
      // current hold/toggle mode; tray is toggle.
      const hold = source === 'fn' ? true : source === 'tray' ? false : settingsRef.current.dictationMode === 'hold';
      if (state === 'pressed') {
        // Native shows the overlay immediately on press; publish live state.
        // Duplicate presses never restart capture; the clip limit bounds a hold.
        overlayOnRef.current = true;
        publishRef.current();
        if (hold) void beginSessionRef.current({ hold: true, overlay: true });
        else void toggleRef.current();
      } else if (hold) {
        // Releasing early must prevent a later recording, not just stop one.
        if (phaseRef.current === 'recording') void stopRef.current();
        else abortPendingRef.current();
      }
    }).then((u) => {
      if (cancelled) u();
      else off = u;
    });
    return () => {
      cancelled = true;
      off?.();
    };
  }, []);

  // Native overlay buttons (main window only).
  const cancelForOverlayRef = useRef(cancel);
  cancelForOverlayRef.current = cancel;
  const copyForOverlayRef = useRef<() => Promise<void>>(async () => undefined);
  const dismissForOverlayRef = useRef(dismissOverlay);
  dismissForOverlayRef.current = dismissOverlay;
  const publishForOverlayRef = useRef(publishSnapshot);
  publishForOverlayRef.current = publishSnapshot;
  const goRef = useRef(go);
  goRef.current = go;
  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    void listenNative<{ action: OverlayAction } | OverlayAction>('overlay-action', (payload) => {
      const action: OverlayAction = typeof payload === 'string' ? payload : payload.action;
      if (action === 'preview') {
        // Native shows the cached snapshot; publish honest current readiness.
        overlayOnRef.current = true;
        publishForOverlayRef.current();
      } else if (action === 'start') {
        overlayOnRef.current = true;
        publishForOverlayRef.current();
        void beginSessionRef.current({ hold: settingsRef.current.dictationMode === 'hold', overlay: true });
      } else if (action === 'stop') void stopRef.current();
      else if (action === 'cancel') void cancelForOverlayRef.current();
      else if (action === 'copy') void copyForOverlayRef.current();
      else if (action === 'open') goRef.current('dictate');
      else if (action === 'dismiss') void dismissForOverlayRef.current();
    }).then((u) => {
      if (cancelled) u();
      else off = u;
    });
    return () => {
      cancelled = true;
      off?.();
    };
  }, []);

  // In-window keys: Escape cancels capture, Mod+Enter toggles it.
  const cancelRef = useRef(cancel);
  cancelRef.current = cancel;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && phaseRef.current === 'recording') {
        if (document.querySelector('[role="dialog"]')) return;
        e.preventDefault();
        void cancelRef.current();
      } else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey) {
        if (isTypingTarget(e.target) && (e.target as HTMLElement).tagName === 'TEXTAREA') return;
        if (document.querySelector('[role="dialog"]')) return;
        e.preventDefault();
        setCommandOpen(false);
        void toggleRef.current();
      }
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [setCommandOpen]);

  // Release the camera when the app unmounts.
  useEffect(
    () => () => {
      const id = clipRef.current;
      if (id) void api.cancelClip(id).catch(() => undefined);
      cameraRef.current?.close();
      cameraRef.current = null;
    },
    [],
  );

  const editDraft = useCallback((text: string) => {
    setDraft((d) => (d ? { ...d, text, dirty: true } : d));
  }, []);

  const saveDraft = useCallback(async () => {
    if (!draft || !draft.entryId) return;
    const entry = historyRef.current.find((h) => h.id === draft.entryId);
    if (!entry) {
      toast.error('That history entry no longer exists');
      return;
    }
    try {
      await saveEntry({ ...entry, text: draft.text, words: countWords(draft.text) });
      setDraft((d) => (d ? { ...d, dirty: false } : d));
      toast.success('Saved to history');
    } catch (e) {
      toast.error('Could not save edit', { description: errorMessage(e) });
    }
  }, [draft, saveEntry]);

  const discardDraft = useCallback(() => setDraft(null), []);

  const copyDraft = useCallback(async () => {
    if (!draft?.text) return;
    try {
      await api.copyText(draft.text);
      toast.success('Copied');
    } catch (e) {
      toast.error('Could not copy', { description: errorMessage(e) });
    }
  }, [draft]);
  copyForOverlayRef.current = copyDraft;

  const pasteDraft = useCallback(async () => {
    if (!draft?.text) return;
    try {
      await api.pasteText(draft.text);
    } catch (e) {
      toast.error('Could not paste', { description: errorMessage(e) });
    }
  }, [draft]);

  const overlaySnapshot = useCallback((): OverlaySnapshot => buildSnapshot(true), [buildSnapshot]);

  const value = useMemo<DictationValue>(
    () => ({
      videoRef,
      phase,
      cameraOpen: phase !== 'idle' && phase !== 'opening',
      devices,
      refreshDevices,
      quality,
      error,
      clearError: () => setError(null),
      elapsed,
      draft,
      editDraft,
      saveDraft,
      discardDraft,
      copyDraft,
      pasteDraft,
      blockedReason,
      openCamera,
      closeCamera,
      start,
      stop,
      cancel,
      toggle,
      overlaySnapshot,
      previewOverlay,
      dismissOverlay,
    }),
    [
      phase, devices, refreshDevices, quality, error, elapsed, draft, editDraft, saveDraft, discardDraft, copyDraft,
      pasteDraft, blockedReason, openCamera, closeCamera, start, stop, cancel, toggle,
      overlaySnapshot, previewOverlay, dismissOverlay,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
