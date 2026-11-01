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
import type { CaptureQuality, HistoryEntry, TranscriptionResult } from './types';
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

export function DictationProvider({ children }: { children: ReactNode }) {
  const { settings, engine, history, saveEntry, setCommandOpen } = useStore();
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

  const openCamera = useCallback(async (): Promise<boolean> => {
    if (cameraRef.current?.isOpen) return true;
    const cam = getCamera();
    if (!cam) {
      setError('The camera view is not ready yet. Switch to Dictate and try again.');
      return false;
    }
    setError(null);
    setPhase('opening');
    try {
      setDevices(await cam.open(settingsRef.current.cameraId || undefined));
      setPhase('ready');
      return true;
    } catch (e) {
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

  const start = useCallback(async () => {
    if (phaseRef.current === 'recording' || phaseRef.current === 'processing' || phaseRef.current === 'opening') return;
    setError(null);
    if (!isDesktop || !engineReadyRef.current) {
      setError(
        !isDesktop
          ? 'Browser preview cannot transcribe. Use the desktop app to dictate.'
          : 'Install the research engine in Models before dictating.',
      );
      return;
    }
    if (!(await openCamera())) return;
    stopRequestedRef.current = false;
    cancelledRef.current = false;
    sessionTextRef.current = false;
    appendErrorRef.current = null;
    try {
      const segmented = settingsRef.current.dictationMode === 'toggle';
      await beginClip(segmented ? () => pauseHandlerRef.current?.() : undefined);
      startedAtRef.current = performance.now();
      setElapsed(0);
      setPhase('recording');
    } catch (e) {
      setError(errorMessage(e));
      setPhase('ready');
    }
  }, [beginClip, openCamera, setPhase]);

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

  // Native global shortcut.
  const toggleRef = useRef(toggle);
  toggleRef.current = toggle;
  const startRef = useRef(start);
  startRef.current = start;
  const stopRef = useRef(stop);
  stopRef.current = stop;
  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    void listenNative<{ state: 'pressed' | 'released' }>('dictation-shortcut', ({ state }) => {
      const hold = settingsRef.current.dictationMode === 'hold';
      // Repeated 'pressed' events (key repeat, tray menu) never stop a hold-mode clip; the clip limit bounds it.
      if (state === 'pressed') void (hold ? startRef.current() : toggleRef.current());
      else if (hold) void stopRef.current();
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

  const pasteDraft = useCallback(async () => {
    if (!draft?.text) return;
    try {
      await api.pasteText(draft.text);
    } catch (e) {
      toast.error('Could not paste', { description: errorMessage(e) });
    }
  }, [draft]);

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
    }),
    [
      phase, devices, refreshDevices, quality, error, elapsed, draft, editDraft, saveDraft, discardDraft, copyDraft,
      pasteDraft, blockedReason, openCamera, closeCamera, start, stop, cancel, toggle,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
