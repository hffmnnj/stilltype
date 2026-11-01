import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { toast } from 'sonner';
import { api, errorMessage, isDesktop, listenNative } from './api';
import type { EngineStatus, HistoryEntry, InstallProgress, Settings, VocabularyWord } from './types';
import { defaultSettings } from './types';

export type View = 'dictate' | 'history' | 'vocabulary' | 'models' | 'settings';
export const VIEWS: View[] = ['dictate', 'history', 'vocabulary', 'models', 'settings'];

interface StoreValue {
  loading: boolean;
  loadError: string | null;
  reload: () => void;
  settings: Settings;
  /** Resolves true only when the change was actually persisted. */
  updateSettings: (patch: Partial<Settings>) => Promise<boolean>;
  settingsError: string | null;
  settingsSaving: boolean;
  history: HistoryEntry[];
  saveEntry: (entry: HistoryEntry) => Promise<HistoryEntry>;
  deleteEntry: (id: string) => Promise<void>;
  clearHistory: () => Promise<void>;
  vocabulary: VocabularyWord[];
  saveWord: (word: VocabularyWord) => Promise<VocabularyWord>;
  deleteWord: (id: string) => Promise<void>;
  engine: EngineStatus | null;
  engineError: string | null;
  refreshEngine: () => Promise<void>;
  installing: boolean;
  installProgress: InstallProgress | null;
  installError: string | null;
  install: () => Promise<void>;
  view: View;
  go: (view: View) => void;
  commandOpen: boolean;
  setCommandOpen: (open: boolean) => void;
  resolvedTheme: 'light' | 'dark';
}

const Ctx = createContext<StoreValue | null>(null);

export function useStore(): StoreValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useStore outside provider');
  return v;
}

function readView(): View {
  const q = new URLSearchParams(window.location.search).get('view');
  return VIEWS.includes(q as View) ? (q as View) : 'dictate';
}

function sortHistory(list: HistoryEntry[]): HistoryEntry[] {
  return list.slice().sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [vocabulary, setVocabulary] = useState<VocabularyWord[]>([]);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installProgress, setInstallProgress] = useState<InstallProgress | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);
  const [view, setView] = useState<View>(readView);
  const [commandOpen, setCommandOpen] = useState(false);
  const [systemDark, setSystemDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve());

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    api
      .getState()
      .then((s) => {
        setSettings({ ...defaultSettings, ...s.settings });
        setHistory(sortHistory(s.history));
        setVocabulary(s.vocabulary);
      })
      .catch((e) => setLoadError(errorMessage(e)))
      .finally(() => setLoading(false));
  }, []);

  const refreshEngine = useCallback(async () => {
    try {
      setEngine(await api.engineStatus());
      setEngineError(null);
    } catch (e) {
      setEngineError(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    load();
    void refreshEngine();
  }, [load, refreshEngine]);

  // theme
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const on = () => setSystemDark(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  const resolvedTheme: 'light' | 'dark' =
    settings.theme === 'system' ? (systemDark ? 'dark' : 'light') : settings.theme;
  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.style.colorScheme = resolvedTheme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolvedTheme === 'dark' ? '#0A0A0A' : '#FFFFFF');
  }, [resolvedTheme]);

  // navigation via ?view=
  useEffect(() => {
    const on = () => setView(readView());
    window.addEventListener('popstate', on);
    return () => window.removeEventListener('popstate', on);
  }, []);
  const go = useCallback((next: View) => {
    const url = new URL(window.location.href);
    if (next === 'dictate') url.searchParams.delete('view');
    else url.searchParams.set('view', next);
    window.history.pushState(null, '', url);
    setView(next);
  }, []);
  useEffect(() => {
    const names: Record<View, string> = {
      dictate: 'Dictate',
      history: 'History',
      vocabulary: 'Vocabulary',
      models: 'Models',
      settings: 'Settings',
    };
    document.title = `${names[view]} · Stilltype`;
  }, [view]);

  // install progress
  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    void listenNative<InstallProgress>('install-progress', (p) => {
      setInstallProgress(p);
      if (p.stage === 'error') setInstallError(p.message);
    }).then((u) => {
      if (cancelled) u();
      else off = u;
    });
    return () => {
      cancelled = true;
      off?.();
    };
  }, []);

  const updateSettings = useCallback(async (patch: Partial<Settings>) => {
    const run = async (): Promise<boolean> => {
      const base = settingsRef.current;
      const next = { ...base, ...patch };
      setSettings(next);
      setSettingsSaving(true);
      try {
        const saved = await api.saveSettings(next);
        setSettings({ ...defaultSettings, ...saved });
        setSettingsError(null);
        return true;
      } catch (e) {
        // Revert only the keys we changed, so a later queued edit is not lost.
        setSettings((cur) => {
          const reverted = { ...cur };
          for (const k of Object.keys(patch) as (keyof Settings)[]) (reverted as Record<string, unknown>)[k] = base[k];
          return reverted;
        });
        const msg = errorMessage(e);
        setSettingsError(msg);
        toast.error('Setting not saved', { description: msg });
        return false;
      } finally {
        setSettingsSaving(false);
      }
    };
    const p = saveQueue.current.then(run, run);
    saveQueue.current = p;
    return p;
  }, []);

  const saveEntry = useCallback(async (entry: HistoryEntry) => {
    const saved = await api.saveEntry(entry);
    setHistory((h) => sortHistory([saved, ...h.filter((x) => x.id !== saved.id)]));
    return saved;
  }, []);
  const deleteEntry = useCallback(async (id: string) => {
    await api.deleteEntry(id);
    setHistory((h) => h.filter((x) => x.id !== id));
  }, []);
  const clearHistory = useCallback(async () => {
    await api.clearHistory();
    setHistory([]);
  }, []);
  const saveWord = useCallback(async (word: VocabularyWord) => {
    const saved = await api.saveWord(word);
    setVocabulary((v) => {
      const i = v.findIndex((x) => x.id === saved.id);
      if (i < 0) return [...v, saved];
      const n = v.slice();
      n[i] = saved;
      return n;
    });
    return saved;
  }, []);
  const deleteWord = useCallback(async (id: string) => {
    await api.deleteWord(id);
    setVocabulary((v) => v.filter((x) => x.id !== id));
  }, []);

  const install = useCallback(async () => {
    setInstalling(true);
    setInstallError(null);
    setInstallProgress({ stage: 'runtime', message: 'Starting', progress: 0 });
    try {
      setEngine(await api.installEngine(true));
    } catch (e) {
      setInstallError(errorMessage(e));
    } finally {
      setInstalling(false);
      void refreshEngine();
    }
  }, [refreshEngine]);

  const value = useMemo<StoreValue>(
    () => ({
      loading,
      loadError,
      reload: load,
      settings,
      updateSettings,
      settingsError,
      settingsSaving,
      history,
      saveEntry,
      deleteEntry,
      clearHistory,
      vocabulary,
      saveWord,
      deleteWord,
      engine,
      engineError,
      refreshEngine,
      installing,
      installProgress,
      installError,
      install,
      view,
      go,
      commandOpen,
      setCommandOpen,
      resolvedTheme,
    }),
    [
      loading, loadError, load, settings, updateSettings, settingsError, settingsSaving, history, saveEntry,
      deleteEntry, clearHistory, vocabulary, saveWord, deleteWord, engine, engineError, refreshEngine, installing,
      installProgress, installError, install, view, go, commandOpen, resolvedTheme,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export { isDesktop };
