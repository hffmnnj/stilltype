import { ChevronDown, Clipboard, Download, Pencil, Search, Star, Trash2, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '../components/ui/dropdown';
import { Button, Empty, IconButton, Notice, PageHeader, Spinner } from '../components/ui/misc';
import { api, errorMessage } from '../lib/api';
import { useStore } from '../lib/store';
import type { HistoryEntry } from '../lib/types';
import { countWords, formatTime, groupByDay } from '../lib/utils';

function EditDialog({ entry, onClose }: { entry: HistoryEntry | null; onClose: () => void }) {
  const { saveEntry } = useStore();
  const [text, setText] = useState(entry?.text ?? '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lastId, setLastId] = useState(entry?.id);
  if (entry && entry.id !== lastId) {
    setLastId(entry.id);
    setText(entry.text);
    setErr(null);
  }
  const save = async () => {
    if (!entry) return;
    setBusy(true);
    setErr(null);
    try {
      await saveEntry({ ...entry, text, words: countWords(text) });
      onClose();
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open={entry !== null} onOpenChange={(o) => !o && onClose()}>
      {entry ? (
        <DialogContent
          title="Edit transcript"
          description="Changes are saved to this entry. The original recognition is kept as it was."
          footer={
            <>
              <DialogClose asChild><Button variant="ghost">Cancel</Button></DialogClose>
              <Button variant="primary" busy={busy} onClick={() => void save()} disabled={!text.trim()}>
                Save
              </Button>
            </>
          }
        >
          <label className="visually-hidden" htmlFor="edit-text">Transcript text</label>
          <textarea id="edit-text" className="field field-large" rows={6} value={text} onChange={(e) => setText(e.target.value)} />
          {err ? <Notice tone="error">{err}</Notice> : null}
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

export function History() {
  const { history, loading, loadError, reload, settings, saveEntry, deleteEntry, clearHistory, go } = useStore();
  const [query, setQuery] = useState('');
  const [favOnly, setFavOnly] = useState(false);
  const [editing, setEditing] = useState<HistoryEntry | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearErr, setClearErr] = useState<string | null>(null);

  const totals = useMemo(
    () => ({
      entries: history.length,
      words: history.reduce((n, h) => n + h.words, 0),
      favorites: history.filter((h) => h.favorite).length,
    }),
    [history],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return history.filter((h) => (!favOnly || h.favorite) && (!q || h.text.toLowerCase().includes(q) || h.raw.toLowerCase().includes(q)));
  }, [history, query, favOnly]);
  const groups = useMemo(() => groupByDay(filtered), [filtered]);

  const copy = async (text: string) => {
    try {
      await api.copyText(text);
      toast.success('Copied');
    } catch (e) {
      toast.error('Could not copy', { description: errorMessage(e) });
    }
  };
  const toggleFav = async (h: HistoryEntry) => {
    try {
      await saveEntry({ ...h, favorite: !h.favorite });
    } catch (e) {
      toast.error('Could not update favorite', { description: errorMessage(e) });
    }
  };
  const remove = async (h: HistoryEntry) => {
    try {
      await deleteEntry(h.id);
      toast('Entry deleted', {
        action: {
          label: 'Undo',
          onClick: () => {
            saveEntry(h).catch((e) => toast.error('Could not restore', { description: errorMessage(e) }));
          },
        },
      });
    } catch (e) {
      toast.error('Could not delete', { description: errorMessage(e) });
    }
  };
  const doExport = async (format: 'json' | 'txt' | 'csv') => {
    try {
      const where = await api.exportData(format);
      toast.success('Exported', { description: where });
    } catch (e) {
      toast.error('Export failed', { description: errorMessage(e) });
    }
  };
  const doClear = async () => {
    setClearing(true);
    setClearErr(null);
    try {
      await clearHistory();
      setConfirmClear(false);
      toast.success('History cleared');
    } catch (e) {
      setClearErr(errorMessage(e));
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="page">
      <PageHeader title="History">
        {history.length > 0 ? (
          <p className="stats mono" aria-label="History totals">
            {totals.entries} {totals.entries === 1 ? 'entry' : 'entries'} · {totals.words.toLocaleString()} words · {totals.favorites} starred
          </p>
        ) : null}
      </PageHeader>

      {!settings.keepHistory ? (
        <Notice tone="warn" action={<Button onClick={() => go('settings')}>Open Settings</Button>}>
          History saving is off. New dictations are not kept.
        </Notice>
      ) : null}

      <div className="toolbar">
        <div className="search">
          <Search size={16} aria-hidden />
          <input
            type="search"
            className="search-input"
            placeholder="Search your dictations"
            aria-label="Search history"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {query ? (
            <IconButton label="Clear search" onClick={() => setQuery('')}><X size={15} aria-hidden /></IconButton>
          ) : null}
        </div>
        <Button onClick={() => setFavOnly((v) => !v)} aria-pressed={favOnly} className={favOnly ? 'is-on' : ''}>
          <Star size={15} fill={favOnly ? 'currentColor' : 'none'} aria-hidden /> Starred
        </Button>
        <Menu>
          <MenuTrigger asChild>
            <Button disabled={history.length === 0}>
              <Download size={15} aria-hidden /> Export <ChevronDown size={14} aria-hidden />
            </Button>
          </MenuTrigger>
          <MenuContent label="Export format">
            <MenuItem onSelect={() => void doExport('json')}>JSON, with vocabulary</MenuItem>
            <MenuItem onSelect={() => void doExport('txt')}>Plain text</MenuItem>
            <MenuItem onSelect={() => void doExport('csv')}>CSV</MenuItem>
          </MenuContent>
        </Menu>
        <Button variant="ghost" className="danger-text" disabled={history.length === 0} onClick={() => setConfirmClear(true)}>
          <Trash2 size={15} aria-hidden /> Clear
        </Button>
      </div>

      {loading ? (
        <div className="list-skeleton"><Spinner label="Loading history" /></div>
      ) : loadError ? (
        <Notice tone="error" action={<Button onClick={reload}>Retry</Button>}>{loadError}</Notice>
      ) : history.length === 0 ? (
        <Empty title="Nothing dictated yet" action={<Button variant="primary" onClick={() => go('dictate')}>Go to Dictate</Button>}>
          Finished dictations are kept here on this device, never sent anywhere.
        </Empty>
      ) : filtered.length === 0 ? (
        <Empty
          title="No matches"
          action={
            <Button onClick={() => { setQuery(''); setFavOnly(false); }}>Clear search and filters</Button>
          }
        >
          Nothing in your history matches that.
        </Empty>
      ) : (
        <div className="history">
          {groups.map((g) => (
            <section key={g.label} aria-label={g.label} className="day">
              <h2 className="day-label">{g.label}</h2>
              <ul className="entries">
                {g.items.map((h) => (
                  <li key={h.id} className="entry">
                    <time className="entry-time mono" dateTime={h.createdAt}>{formatTime(h.createdAt)}</time>
                    <div className="entry-body">
                      <p className="entry-text">{h.text}</p>
                      {h.raw.trim() && h.raw.trim() !== h.text.trim() ? (
                        <details className="raw">
                          <summary>Original recognition</summary>
                          <p>{h.raw}</p>
                        </details>
                      ) : null}
                      <p className="entry-meta mono">{h.words} words{h.durationMs ? ` · ${(h.durationMs / 1000).toFixed(1)}s` : ''}</p>
                    </div>
                    <div className="entry-actions">
                      <IconButton label={h.favorite ? 'Remove star' : 'Star'} pressed={h.favorite} onClick={() => void toggleFav(h)} className={h.favorite ? 'starred' : ''}>
                        <Star size={16} fill={h.favorite ? 'currentColor' : 'none'} aria-hidden />
                      </IconButton>
                      <IconButton label="Copy" onClick={() => void copy(h.text)}><Clipboard size={16} aria-hidden /></IconButton>
                      <IconButton label="Edit" onClick={() => setEditing(h)}><Pencil size={16} aria-hidden /></IconButton>
                      <IconButton label="Delete" onClick={() => void remove(h)}><Trash2 size={16} aria-hidden /></IconButton>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}

      <EditDialog entry={editing} onClose={() => setEditing(null)} />

      <Dialog open={confirmClear} onOpenChange={(o) => { setConfirmClear(o); if (!o) setClearErr(null); }}>
        <DialogContent
          title="Clear all history?"
          description={`This permanently removes ${totals.entries} ${totals.entries === 1 ? 'entry' : 'entries'} from this device. Export first if you want a copy. Vocabulary is kept.`}
          footer={
            <>
              <DialogClose asChild><Button variant="ghost">Cancel</Button></DialogClose>
              <Button variant="danger" busy={clearing} onClick={() => void doClear()}>Clear history</Button>
            </>
          }
        >
          {clearErr ? <Notice tone="error">{clearErr}</Notice> : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
