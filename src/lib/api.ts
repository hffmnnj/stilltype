import type {
  AppState,
  CapturedFrame,
  DesktopApi,
  EngineStatus,
  HistoryEntry,
  InstallProgress,
  Settings,
  TranscriptionResult,
  VocabularyWord,
} from './types';
import { defaultSettings } from './types';

export const isDesktop: boolean =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export const DESKTOP_REQUIRED =
  'Transcription needs the Stilltype desktop app. The browser preview can only store settings, history and vocabulary.';

const KEY = {
  settings: 'stilltype.settings',
  history: 'stilltype.history',
  vocabulary: 'stilltype.vocabulary',
} as const;

/* ---------- browser preview storage ---------- */

function readJson<T>(key: string, fallback: T): T {
  const raw = window.localStorage.getItem(key);
  if (raw === null) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    // Never overwrite data we cannot parse; surface it instead.
    throw new Error(`Saved data under “${key}” is unreadable. It was left untouched.`);
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    throw new Error(`Could not save to browser storage: ${errorMessage(e)}`);
  }
}

export function errorMessage(e: unknown): string {
  if (typeof e === 'string') return e;
  if (e instanceof Error) return e.message;
  if (e && typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message);
  return 'Something went wrong.';
}

function upsert<T extends { id: string }>(list: T[], item: T, newestFirst = false): T[] {
  const i = list.findIndex((x) => x.id === item.id);
  if (i >= 0) {
    const next = list.slice();
    next[i] = item;
    return next;
  }
  return newestFirst ? [item, ...list] : [...list, item];
}

function download(name: string, mime: string, body: string): void {
  const url = URL.createObjectURL(new Blob([body], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(v: string | number | boolean): string {
  let s = String(v);
  // Neutralise spreadsheet formulas, matching the native export.
  if (/^[=+\-@\t\r\n]/.test(s)) s = `'${s}`;
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function buildExport(format: 'json' | 'txt' | 'csv', state: AppState): { name: string; mime: string; body: string } {
  const stamp = new Date().toISOString().slice(0, 10);
  const { history, vocabulary } = state;
  if (format === 'json') {
    return {
      name: `stilltype-${stamp}.json`,
      mime: 'application/json',
      body: JSON.stringify({ history, vocabulary }, null, 2),
    };
  }
  if (format === 'txt') {
    const body = history
      .map((h) => `${new Date(h.createdAt).toLocaleString()}${h.favorite ? ' ★' : ''}\n${h.text}`)
      .join('\n\n');
    return { name: `stilltype-${stamp}.txt`, mime: 'text/plain', body };
  }
  const head = 'id,createdAt,text,raw,favorite,words,durationMs,inferenceMs';
  const rows = history.map((h) =>
    [h.id, h.createdAt, h.text, h.raw, h.favorite, h.words, h.durationMs, h.inferenceMs].map(csvCell).join(','),
  );
  return { name: `stilltype-${stamp}.csv`, mime: 'text/csv', body: [head, ...rows].join('\n') };
}

const browserApi: DesktopApi = {
  async getState() {
    const stored = readJson<Partial<Settings>>(KEY.settings, {});
    return {
      settings: { ...defaultSettings, ...stored },
      history: readJson<HistoryEntry[]>(KEY.history, []),
      vocabulary: readJson<VocabularyWord[]>(KEY.vocabulary, []),
    };
  },
  async saveSettings(settings) {
    writeJson(KEY.settings, settings);
    return settings;
  },
  async saveEntry(entry) {
    writeJson(KEY.history, upsert(readJson<HistoryEntry[]>(KEY.history, []), entry, true));
    return entry;
  },
  async deleteEntry(id) {
    writeJson(KEY.history, readJson<HistoryEntry[]>(KEY.history, []).filter((e) => e.id !== id));
  },
  async clearHistory() {
    writeJson(KEY.history, []);
  },
  async saveWord(word) {
    writeJson(KEY.vocabulary, upsert(readJson<VocabularyWord[]>(KEY.vocabulary, []), word));
    return word;
  },
  async deleteWord(id) {
    writeJson(KEY.vocabulary, readJson<VocabularyWord[]>(KEY.vocabulary, []).filter((w) => w.id !== id));
  },
  async engineStatus(): Promise<EngineStatus> {
    return {
      runtimeInstalled: false,
      modelInstalled: false,
      ready: false,
      device: 'Browser preview',
      error: null,
      dataDir: '',
      uvAvailable: false,
    };
  },
  async installEngine() {
    throw new Error('Installing the research engine needs the Stilltype desktop app.');
  },
  async startClip() {
    throw new Error(DESKTOP_REQUIRED);
  },
  async appendFrames() {
    throw new Error(DESKTOP_REQUIRED);
  },
  async transcribe(): Promise<TranscriptionResult> {
    throw new Error(DESKTOP_REQUIRED);
  },
  async cancelClip() {
    /* nothing running in the browser */
  },
  async copyText(text) {
    if (!navigator.clipboard) throw new Error('Clipboard access is not available in this browser.');
    await navigator.clipboard.writeText(text);
  },
  async pasteText() {
    throw new Error('Pasting into other apps needs the desktop app.');
  },
  async ollamaModels() {
    throw new Error('Listing Ollama models needs the desktop app, which talks to 127.0.0.1 for you.');
  },
  async exportData(format) {
    const state = await browserApi.getState();
    const out = buildExport(format, state);
    download(out.name, out.mime, out.body);
    return out.name;
  },
};

/* ---------- desktop (Tauri) ---------- */

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
  return tauriInvoke<T>(cmd, args);
}

const desktopApi: DesktopApi = {
  getState: () => invoke<AppState>('get_state'),
  saveSettings: (settings) => invoke<Settings>('save_settings', { settings }),
  saveEntry: (entry) => invoke<HistoryEntry>('save_entry', { entry }),
  deleteEntry: (id) => invoke<void>('delete_entry', { id }),
  clearHistory: () => invoke<void>('clear_history'),
  saveWord: (word) => invoke<VocabularyWord>('save_word', { word }),
  deleteWord: (id) => invoke<void>('delete_word', { id }),
  engineStatus: () => invoke<EngineStatus>('engine_status'),
  installEngine: (acceptLicense) => invoke<EngineStatus>('install_engine', { acceptLicense }),
  startClip: () => invoke<string>('start_clip'),
  appendFrames: (clipId: string, frames: CapturedFrame[]) => invoke<void>('append_frames', { clipId, frames }),
  transcribe: (clipId, final) => invoke<TranscriptionResult>('transcribe', { clipId, final }),
  cancelClip: (clipId) => invoke<void>('cancel_clip', { clipId }),
  copyText: (text) => invoke<void>('copy_text', { text }),
  pasteText: (text) => invoke<void>('paste_text', { text }),
  ollamaModels: () => invoke<string[]>('ollama_models'),
  async exportData(format) {
    // Native returns the rendered file contents; saving is done here.
    const body = await invoke<string>('export_data', { format });
    const stamp = new Date().toISOString().slice(0, 10);
    const mime = { json: 'application/json', txt: 'text/plain', csv: 'text/csv' }[format];
    const name = `stilltype-${stamp}.${format}`;
    download(name, mime, body);
    return name;
  },
};

export const api: DesktopApi = isDesktop ? desktopApi : browserApi;

/** Subscribe to a native event. Returns an unsubscribe function; no-op in the browser. */
export async function listenNative<T>(event: string, handler: (payload: T) => void): Promise<() => void> {
  if (!isDesktop) return () => undefined;
  const { listen } = await import('@tauri-apps/api/event');
  return listen<T>(event, (e) => handler(e.payload));
}

export type { InstallProgress };
