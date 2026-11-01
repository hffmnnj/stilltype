import { BookA, Cpu, History, ScanFace, Search, Settings } from 'lucide-react';
import type { ReactNode } from 'react';
import { useDictation } from '../lib/dictation';
import { isDesktop } from '../lib/api';
import { useStore, type View } from '../lib/store';
import { modKey } from '../lib/utils';
import { Keycaps, Mark } from './ui/misc';

const ITEMS: { view: View; label: string; icon: ReactNode }[] = [
  { view: 'dictate', label: 'Dictate', icon: <ScanFace size={18} aria-hidden /> },
  { view: 'history', label: 'History', icon: <History size={18} aria-hidden /> },
  { view: 'vocabulary', label: 'Vocabulary', icon: <BookA size={18} aria-hidden /> },
  { view: 'models', label: 'Models', icon: <Cpu size={18} aria-hidden /> },
  { view: 'settings', label: 'Settings', icon: <Settings size={18} aria-hidden /> },
];

export function Rail() {
  const { view, go, setCommandOpen, engine } = useStore();
  const d = useDictation();
  return (
    <>
      <aside className="rail">
        <div className="brand">
          <Mark size={30} />
          <span className="wordmark">Stilltype</span>
        </div>
        <nav aria-label="Primary" className="nav">
          {ITEMS.map((i) => (
            <button key={i.view} type="button" className="nav-item" aria-current={view === i.view ? 'page' : undefined} onClick={() => go(i.view)}>
              {i.icon}
              <span>{i.label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-foot">
          <button type="button" className="nav-item cmd-hint" onClick={() => setCommandOpen(true)}>
            <Search size={16} aria-hidden />
            <span>Commands</span>
            <Keycaps keys={[modKey, 'K']} />
          </button>
          <div className="rail-status" role="status">
            <p><span className="dot" data-on={d.cameraOpen || undefined} aria-hidden />Camera {d.cameraOpen ? 'on' : 'off'}</p>
            <p><span className="dot" data-on={engine?.ready || undefined} aria-hidden />Engine {engine?.ready ? 'ready' : 'not ready'}</p>
            {!isDesktop ? <p className="preview-tag">Browser preview</p> : null}
          </div>
        </div>
      </aside>
      <nav aria-label="Primary" className="tabbar">
        {ITEMS.map((i) => (
          <button key={i.view} type="button" className="tab-item" aria-current={view === i.view ? 'page' : undefined} onClick={() => go(i.view)}>
            {i.icon}
            <span>{i.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}
