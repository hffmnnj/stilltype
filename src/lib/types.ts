export type Theme = 'system' | 'light' | 'dark';
export type DictationMode = 'hold' | 'toggle';
export interface Settings {
  theme: Theme;
  cameraId: string;
  mirrored: boolean;
  showPreview: boolean;
  shortcut: string;
  dictationMode: DictationMode;
  autoPaste: boolean;
  keepHistory: boolean;
  retentionDays: number;
  cleanupEnabled: boolean;
  cleanupModel: string;
  pauseSeconds: number;
  beamSize: number;
  inferenceDevice: 'auto' | 'gpu' | 'cpu';
  maxClipSeconds: number;
  context: string;
}
export interface HistoryEntry {
  id: string;
  text: string;
  raw: string;
  createdAt: string;
  durationMs: number;
  inferenceMs: number;
  favorite: boolean;
  words: number;
}
export interface VocabularyWord {
  id: string;
  word: string;
  replacement: string;
  createdAt: string;
}
export interface AppState {
  settings: Settings;
  history: HistoryEntry[];
  vocabulary: VocabularyWord[];
}
export interface EngineStatus {
  runtimeInstalled: boolean;
  modelInstalled: boolean;
  ready: boolean;
  device: string;
  error: string | null;
  dataDir: string;
  uvAvailable: boolean;
}
export interface InstallProgress {
  stage: 'runtime' | 'download' | 'verify' | 'ready' | 'error';
  message: string;
  progress: number;
}
export interface CapturedFrame {
  jpeg: string;
  timestamp: number;
  landmarks: [number, number][] | null;
}
export interface TranscriptionResult {
  text: string;
  raw: string;
  durationMs: number;
  inferenceMs: number;
  frames: number;
  error: string | null;
}
export interface CaptureQuality {
  faceDetected: boolean;
  brightness: number;
  motion: number;
  fps: number;
  hint: string;
}
export const defaultSettings: Settings = {
  theme: 'system', cameraId: '', mirrored: true, showPreview: true,
  shortcut: 'CommandOrControl+Shift+Space', dictationMode: 'toggle',
  autoPaste: false, keepHistory: true, retentionDays: 30,
  cleanupEnabled: false, cleanupModel: 'qwen3.5:4b',
  pauseSeconds: 0.7, beamSize: 5, inferenceDevice: 'auto', maxClipSeconds: 30, context: '',
};
/** Native command arguments and worker requests use camelCase payload fields. */
export interface DesktopApi {
  getState(): Promise<AppState>;
  saveSettings(settings: Settings): Promise<Settings>;
  saveEntry(entry: HistoryEntry): Promise<HistoryEntry>;
  deleteEntry(id: string): Promise<void>;
  clearHistory(): Promise<void>;
  saveWord(word: VocabularyWord): Promise<VocabularyWord>;
  deleteWord(id: string): Promise<void>;
  engineStatus(): Promise<EngineStatus>;
  installEngine(acceptLicense: boolean): Promise<EngineStatus>;
  startClip(): Promise<string>;
  appendFrames(clipId: string, frames: CapturedFrame[]): Promise<void>;
  transcribe(clipId: string, final: boolean): Promise<TranscriptionResult>;
  cancelClip(clipId: string): Promise<void>;
  copyText(text: string): Promise<void>;
  pasteText(text: string): Promise<void>;
  ollamaModels(): Promise<string[]>;
  exportData(format: 'json' | 'txt' | 'csv'): Promise<string>;
}
