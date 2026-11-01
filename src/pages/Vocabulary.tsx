import { ArrowRight, Pencil, Plus, Trash2 } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { toast } from 'sonner';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Button, Empty, IconButton, Notice, PageHeader, Spinner } from '../components/ui/misc';
import { errorMessage } from '../lib/api';
import { useStore } from '../lib/store';
import type { VocabularyWord } from '../lib/types';
import { newId } from '../lib/utils';

function WordFields({
  word,
  replacement,
  setWord,
  setReplacement,
  idPrefix,
}: {
  word: string;
  replacement: string;
  setWord: (v: string) => void;
  setReplacement: (v: string) => void;
  idPrefix: string;
}) {
  return (
    <>
      <div className="field-group">
        <label htmlFor={`${idPrefix}-w`} className="field-label">Word as heard</label>
        <input id={`${idPrefix}-w`} className="field" value={word} onChange={(e) => setWord(e.target.value)} autoComplete="off" spellCheck={false} maxLength={80} />
      </div>
      <div className="field-group">
        <label htmlFor={`${idPrefix}-r`} className="field-label">Write it as <span className="muted">(optional)</span></label>
        <input id={`${idPrefix}-r`} className="field" value={replacement} onChange={(e) => setReplacement(e.target.value)} autoComplete="off" spellCheck={false} maxLength={80} />
      </div>
    </>
  );
}

export function Vocabulary() {
  const { vocabulary, saveWord, deleteWord, loading, loadError, reload } = useStore();
  const [word, setWord] = useState('');
  const [replacement, setReplacement] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [editing, setEditing] = useState<VocabularyWord | null>(null);
  const [eWord, setEWord] = useState('');
  const [eRep, setERep] = useState('');
  const [eErr, setEErr] = useState<string | null>(null);
  const [eBusy, setEBusy] = useState(false);

  const duplicate = (w: string, ignoreId?: string) =>
    vocabulary.some((v) => v.id !== ignoreId && v.word.trim().toLowerCase() === w.trim().toLowerCase());

  const add = async (ev: FormEvent) => {
    ev.preventDefault();
    const w = word.trim();
    if (!w) return;
    if (duplicate(w)) {
      setErr(`“${w}” is already in your vocabulary. Edit it instead.`);
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await saveWord({ id: newId(), word: w, replacement: replacement.trim(), createdAt: new Date().toISOString() });
      setWord('');
      setReplacement('');
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const openEdit = (v: VocabularyWord) => {
    setEditing(v);
    setEWord(v.word);
    setERep(v.replacement);
    setEErr(null);
  };
  const saveEdit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (!editing) return;
    const w = eWord.trim();
    if (!w) return;
    if (duplicate(w, editing.id)) {
      setEErr(`“${w}” is already in your vocabulary.`);
      return;
    }
    setEBusy(true);
    try {
      await saveWord({ ...editing, word: w, replacement: eRep.trim() });
      setEditing(null);
    } catch (e) {
      setEErr(errorMessage(e));
    } finally {
      setEBusy(false);
    }
  };
  const remove = async (v: VocabularyWord) => {
    try {
      await deleteWord(v.id);
      toast('Word removed', {
        action: {
          label: 'Undo',
          onClick: () => {
            saveWord(v).catch((e) => toast.error('Could not restore', { description: errorMessage(e) }));
          },
        },
      });
    } catch (e) {
      toast.error('Could not remove', { description: errorMessage(e) });
    }
  };

  return (
    <div className="page narrow">
      <PageHeader title="Vocabulary">
        {vocabulary.length > 0 ? <p className="stats mono">{vocabulary.length} {vocabulary.length === 1 ? 'word' : 'words'}</p> : null}
      </PageHeader>
      <p className="lede">Names, jargon and spellings the model tends to miss. Entries save the moment you add them.</p>

      <form className="add-word" onSubmit={(e) => void add(e)} aria-label="Add a word">
        <WordFields word={word} replacement={replacement} setWord={setWord} setReplacement={setReplacement} idPrefix="new" />
        <Button type="submit" variant="primary" busy={busy} disabled={!word.trim()}>
          <Plus size={16} aria-hidden /> Add
        </Button>
      </form>
      {err ? <Notice tone="error">{err}</Notice> : null}

      {loading ? (
        <div className="list-skeleton"><Spinner label="Loading vocabulary" /></div>
      ) : loadError ? (
        <Notice tone="error" action={<Button onClick={reload}>Retry</Button>}>{loadError}</Notice>
      ) : vocabulary.length === 0 ? (
        <Empty title="No words yet">
          Add a word above. With a replacement, Stilltype writes that spelling instead. Without one, the word is passed along as a known term.
        </Empty>
      ) : (
        <ul className="words">
          {vocabulary.map((v) => (
            <li key={v.id} className="word">
              <span className="word-main">{v.word}</span>
              {v.replacement ? (
                <>
                  <ArrowRight size={14} className="muted" aria-label="becomes" />
                  <span className="word-rep">{v.replacement}</span>
                </>
              ) : (
                <span className="muted small">known term</span>
              )}
              <span className="push-right entry-actions">
                <IconButton label={`Edit ${v.word}`} onClick={() => openEdit(v)}><Pencil size={16} aria-hidden /></IconButton>
                <IconButton label={`Delete ${v.word}`} onClick={() => void remove(v)}><Trash2 size={16} aria-hidden /></IconButton>
              </span>
            </li>
          ))}
        </ul>
      )}

      <Dialog open={editing !== null} onOpenChange={(o) => !o && setEditing(null)}>
        <DialogContent title="Edit word" description="Changes save when you press Save.">
          <form onSubmit={(e) => void saveEdit(e)} className="dialog-form">
            <WordFields word={eWord} replacement={eRep} setWord={setEWord} setReplacement={setERep} idPrefix="edit" />
            {eErr ? <Notice tone="error">{eErr}</Notice> : null}
            <div className="dialog-foot flush">
              <DialogClose asChild><Button variant="ghost">Cancel</Button></DialogClose>
              <Button type="submit" variant="primary" busy={eBusy} disabled={!eWord.trim()}>Save</Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
