import { useEffect } from 'react';
import { Toaster } from 'sonner';
import { CommandMenu } from './components/CommandMenu';
import { Rail } from './components/Rail';
import { DictationProvider } from './lib/dictation';
import { StoreProvider, useStore } from './lib/store';
import { Dictate } from './pages/Dictate';
import { History } from './pages/History';
import { Models } from './pages/Models';
import { SettingsPage } from './pages/Settings';
import { Vocabulary } from './pages/Vocabulary';

function Shell() {
  const { view, setCommandOpen, resolvedTheme } = useStore();
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setCommandOpen(true);
      }
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [setCommandOpen]);

  return (
    <div className="app">
      <a className="skip" href="#main">Skip to content</a>
      <Rail />
      <main id="main" className="main" tabIndex={-1}>
        {/* Dictate stays mounted so the camera and shortcut keep working on other pages. */}
        <div hidden={view !== 'dictate'}><Dictate /></div>
        {view === 'history' ? <History /> : null}
        {view === 'vocabulary' ? <Vocabulary /> : null}
        {view === 'models' ? <Models /> : null}
        {view === 'settings' ? <SettingsPage /> : null}
      </main>
      <CommandMenu />
      <Toaster theme={resolvedTheme} position="bottom-right" toastOptions={{ className: 'toast' }} />
    </div>
  );
}

export default function App() {
  return (
    <StoreProvider>
      <DictationProvider>
        <Shell />
      </DictationProvider>
    </StoreProvider>
  );
}
