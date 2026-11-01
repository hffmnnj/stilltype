import { AlertTriangle, CheckCircle2, Download, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Button, Notice, PageHeader, Segmented, Spinner } from '../components/ui/misc';
import { Toggle } from '../components/ui/switch';
import { api, errorMessage, isDesktop, listenNative } from '../lib/api';
import type { InstallProgress } from '../lib/api';
import { useStore } from '../lib/store';
import type { CleanupMode, CleanupModel, CleanupStatus, Settings } from '../lib/types';

function Row({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="kv">
      <dt>{label}</dt>
      <dd>
        {ok !== undefined ? (
          ok ? <CheckCircle2 size={15} className="ok" aria-hidden /> : <AlertTriangle size={15} className="warn" aria-hidden />
        ) : null}
        <span>{value}</span>
      </dd>
    </div>
  );
}

function Engine() {
  const { engine, engineError, refreshEngine, installing, installProgress, installError, install } = useStore();
  const [consent, setConsent] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const pct = Math.round((installProgress?.progress ?? 0) * 100);
  const ready = Boolean(engine?.ready);

  return (
    <section className="block" aria-labelledby="eng-h">
      <div className="block-head">
        <h2 id="eng-h" className="section-title">Research engine</h2>
        <span className={`status ${ready ? 'status-ok' : 'status-idle'}`}>{!engine ? 'Checking' : ready ? 'Ready' : 'Not installed'}</span>
      </div>
      <p className="lede">
        USR 2.0 Base+ (2026). Reads lip movement locally on your GPU, or your CPU if you choose.
      </p>

      {engineError ? (
        <Notice tone="error" action={<Button onClick={() => void refreshEngine()}>Retry</Button>}>{engineError}</Notice>
      ) : null}

      {!isDesktop ? (
        <Notice tone="warn">
          Browser preview. The engine installs and runs in the desktop app only.
        </Notice>
      ) : null}

      <dl className="kvs">
        <Row label="Python runtime" value={engine?.runtimeInstalled ? 'Installed' : 'Not installed'} ok={engine ? engine.runtimeInstalled : undefined} />
        <Row label="Model weights" value={engine?.modelInstalled ? 'Installed' : 'Not installed'} ok={engine ? engine.modelInstalled : undefined} />
        <Row label="Compute" value={engine?.device || 'Unknown'} />
        <Row label="Package manager (uv)" value={engine ? (engine.uvAvailable ? 'Found' : 'Not found — install uv first') : '…'} />
        {engine && !engine.uvAvailable ? (
          <p className="muted small">
            The installer needs uv already installed.{' '}
            <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank" rel="noreferrer">
              uv installation guide
            </a>
          </p>
        ) : null}
        {engine?.dataDir ? <Row label="Stored in" value={engine.dataDir} /> : null}
      </dl>

      {engine?.error ? <Notice tone="error">{engine.error}</Notice> : null}

      <div className="license">
        <span className="license-tag">License</span>
        <span>Model weights follow the upstream USR 2.0 terms, separate from the app’s MIT license. Review the upstream terms for your intended use.{' '}
          <a href="https://github.com/ahaliassos/usr2" target="_blank" rel="noreferrer">
            Upstream USR 2.0 repository
          </a>
        </span>
      </div>

      {installing || (installProgress && installProgress.stage !== 'ready' && installProgress.stage !== 'error' && !ready) ? (
        <div className="progress-block">
          <div className="progress-line">
            <span>{installProgress?.message || 'Working'}</span>
            <span className="mono">{pct}%</span>
          </div>
          <progress className="progress" max={100} value={pct} aria-label="Install progress" />
        </div>
      ) : null}

      {installError ? (
        <Notice tone="error" action={<Button onClick={() => { setAgreed(false); setConsent(true); }}>Try again</Button>}>
          {installError}
        </Notice>
      ) : null}

      {!ready ? (
        <div className="actions">
          <Button variant="primary" disabled={!isDesktop || installing || !engine} busy={installing} onClick={() => { setAgreed(false); setConsent(true); }}>
            <Download size={16} aria-hidden /> {installing ? 'Installing…' : 'Download and install'}
          </Button>
          <span className="muted small">About 1.4 GB of weights, plus a Python runtime.</span>
        </div>
      ) : (
        <div className="actions">
          <Button onClick={() => void refreshEngine()}><RefreshCw size={15} aria-hidden /> Check again</Button>
        </div>
      )}

      <Dialog open={consent} onOpenChange={setConsent}>
        <DialogContent
          title="Download the research engine?"
          description="This downloads about 1.4 GB of model weights and sets up a Python runtime on this device. Nothing is uploaded."
          footer={
            <>
              <DialogClose asChild><Button variant="ghost">Not now</Button></DialogClose>
              <Button
                variant="primary"
                disabled={!agreed}
                onClick={() => {
                  setConsent(false);
                  void install();
                }}
              >
                Accept and download
              </Button>
            </>
          }
        >
          <label className="consent">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
            <span>
              I understand this downloads about 1.4 GB of model weights, and that the weights follow the upstream USR 2.0 model terms, separate from the app’s MIT license.
            </span>
          </label>
        </DialogContent>
      </Dialog>
    </section>
  );
}

function Compute() {
  const { settings, updateSettings } = useStore();
  return (
    <section className="block" aria-labelledby="dev-h">
      <h2 id="dev-h" className="section-title">Compute</h2>
      <div className="row">
        <div className="row-text">
          <p className="row-label">Run on</p>
          <p className="row-hint">Auto prefers your GPU and falls back to the CPU. CPU is slower.</p>
        </div>
        <Segmented<Settings['inferenceDevice']>
          label="Compute device"
          value={settings.inferenceDevice}
          onChange={(v) => void updateSettings({ inferenceDevice: v })}
          options={[
            { value: 'auto', label: 'Auto' },
            { value: 'gpu', label: 'GPU' },
            { value: 'cpu', label: 'CPU' },
          ]}
        />
      </div>
    </section>
  );
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return 'unknown size';
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(0)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

const modelChoices: { value: CleanupModel; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'qwen3.5-0.8b-2026-q4', label: 'Qwen3.5 0.8B' },
  { value: 'qwen3.5-2b-2026-q4', label: 'Qwen3.5 2B' },
  { value: 'qwen3.5-4b-2026-q4', label: 'Qwen3.5 4B' },
];

function modelLabel(status: CleanupStatus | null, id: string): string {
  return status?.models.find((m) => m.id === id)?.label ?? modelChoices.find((m) => m.value === id)?.label ?? id;
}

function Cleanup() {
  const { settings, updateSettings } = useStore();
  const [status, setStatus] = useState<CleanupStatus | null>(null);
  const [busy, setBusy] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState<InstallProgress | null>(null);
  const [consent, setConsent] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const [target, setTarget] = useState<{ label: string; bytes: number } | null>(null);
  const token = useRef(0);

  const refresh = useCallback(async () => {
    const mine = ++token.current;
    setBusy(true);
    setErr(null);
    try {
      const next = await api.cleanupStatus();
      if (mine === token.current) setStatus(next);
    } catch (e) {
      if (mine === token.current) setErr(errorMessage(e));
    } finally {
      if (mine === token.current) setBusy(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Status always describes the saved selection, so wait for the save before asking again.
  const chooseModel = async (value: CleanupModel) => {
    if (value === settings.cleanupModel) return;
    token.current++;
    setStatus(null);
    setBusy(true);
    setProgress(null);
    setErr(null);
    setSwitching(true);
    try {
      await updateSettings({ cleanupModel: value });
    } finally {
      setSwitching(false);
      void refresh();
    }
  };

  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    void listenNative<InstallProgress>('cleanup-install-progress', (p) => {
      setProgress(p);
      if (p.stage === 'error') setErr(p.message);
    }).then((u) => {
      if (cancelled) u();
      else off = u;
    });
    return () => {
      cancelled = true;
      off?.();
    };
  }, []);

  const install = async () => {
    setInstalling(true);
    setErr(null);
    setProgress({ stage: 'runtime', message: 'Starting', progress: 0 });
    try {
      const next = await api.installCleanup(true);
      token.current++;
      setStatus(next);
      setBusy(false);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setInstalling(false);
      setConsent(false);
    }
  };

  const checking = busy || switching || !status;
  const ready = !checking && Boolean(status?.ready);
  const pct = Math.round((progress?.progress ?? 0) * 100);
  const locked = installing || switching;
  const modelName = status ? modelLabel(status, status.modelId) : '';
  const recommendedName = status ? modelLabel(status, status.recommendedModelId) : '';
  const sizeKnown = Boolean(status && status.totalBytes > 0);
  const choiceText = (c: { value: CleanupModel; label: string }) => {
    const m = status?.models.find((x) => x.id === c.value);
    if (c.value === 'auto') return isDesktop && status && !checking ? `Auto · ${recommendedName} for this device` : 'Auto';
    return m ? `${c.label} · ${formatBytes(m.totalBytes)}${m.installed ? ' · installed' : ''}` : c.label;
  };
  const openConsent = () => {
    if (!status) return;
    setTarget({ label: modelName, bytes: status.totalBytes });
    setAgreed(false);
    setConsent(true);
  };
  const intent = settings.cleanupMode === 'intent';

  return (
    <section className="block" aria-labelledby="cleanup-h">
      <div className="block-head">
        <h2 id="cleanup-h" className="section-title">Transcript cleanup</h2>
        <span className={`status ${ready ? 'status-ok' : 'status-idle'}`}>{checking ? 'Checking' : ready ? 'Ready' : 'Not installed'}</span>
      </div>
      <p className="lede">
        Runs a small Qwen3.5 text model on this device after recognition. The original recognition is always kept.
      </p>

      {!isDesktop ? (
        <Notice tone="warn">
          Browser preview. The cleanup model installs and runs in the desktop app only.
        </Notice>
      ) : null}

      <div className="row">
        <div className="row-text">
          <label htmlFor="cleanup-model" className="row-label">Model</label>
          <p className="row-hint">Auto picks by memory and uses an installed tier before downloading. This is capacity, not accuracy.</p>
        </div>
        <select
          id="cleanup-model"
          className="field select"
          value={settings.cleanupModel}
          disabled={locked}
          onChange={(e) => void chooseModel(e.target.value as CleanupModel)}
        >
          {modelChoices.map((c) => <option key={c.value} value={c.value}>{choiceText(c)}</option>)}
        </select>
      </div>

      <div className="row">
        <div className="row-text">
          <p className="row-label">Mode</p>
          <p className="row-hint">
            {intent
              ? 'Experimental. May change wording, so review the result. Falls back to the recognized text with a warning when edits go too far.'
              : 'Casing and punctuation only. Every word is kept.'}
          </p>
        </div>
        <Segmented<CleanupMode>
          label="Cleanup mode"
          value={settings.cleanupMode}
          disabled={installing}
          onChange={(v) => void updateSettings({ cleanupMode: v })}
          options={[{ value: 'format', label: 'Format' }, { value: 'intent', label: 'Intent' }]}
        />
      </div>

      <dl className="kvs">
        <Row label="Engine" value={status && !checking ? (status.runtimeBundled ? 'Bundled (llama.cpp)' : 'Not bundled') : '…'} ok={status && !checking ? status.runtimeBundled : undefined} />
        <Row label="Selected model" value={checking || !status ? '…' : `${modelName}${isDesktop && status.modelId !== status.recommendedModelId ? ` (recommended: ${recommendedName})` : ''}`} ok={checking || !status ? undefined : status.modelInstalled} />
        <Row label="Download size" value={checking || !status ? '…' : sizeKnown ? formatBytes(status.totalBytes) : 'Unavailable'} />
        <Row label="Compute" value={checking || !status ? '…' : status.device} />
      </dl>

      {!checking && status?.error ? <Notice tone="error">{status.error}</Notice> : null}
      {err ? (
        <Notice tone="error" action={<Button onClick={() => void refresh()} disabled={locked}>Retry</Button>}>{err}</Notice>
      ) : null}

      {installing ? (
        <div className="progress-block">
          <div className="progress-line">
            <span>{progress?.message || 'Working'}</span>
            <span className="mono">{pct}%</span>
          </div>
          <progress className="progress" max={100} value={pct} aria-label="Cleanup install progress" />
        </div>
      ) : null}

      <Toggle
        checked={settings.cleanupEnabled}
        onCheckedChange={(v) => void updateSettings({ cleanupEnabled: v })}
        disabled={locked || (!ready && !settings.cleanupEnabled)}
        label={intent ? 'Correct intent after dictation' : 'Tidy transcripts after dictation'}
        hint={
          checking
            ? 'Checking the selected model.'
            : ready
              ? intent ? 'Runs after final recognition and keeps the original recognition.' : 'Fixes casing and punctuation locally after each transcript.'
              : settings.cleanupEnabled ? 'The selected model is not installed, so transcripts stay as recognized.' : 'Download the selected model to turn this on.'
        }
      />

      {!ready ? (
        <div className="actions">
          <Button variant="primary" disabled={!isDesktop || checking || locked || !sizeKnown} busy={installing} onClick={openConsent}>
            <Download size={16} aria-hidden /> {installing ? 'Downloading…' : `Download ${modelName || 'model'}`}
          </Button>
          {sizeKnown && status ? <span className="muted small">About {formatBytes(status.totalBytes)}.</span> : null}
        </div>
      ) : (
        <div className="actions">
          <Button onClick={() => void refresh()} busy={busy} disabled={locked}>
            {!busy ? <RefreshCw size={15} aria-hidden /> : null} Check again
          </Button>
        </div>
      )}

      <Dialog open={consent} onOpenChange={setConsent}>
        <DialogContent
          title={`Download ${target?.label ?? 'the cleanup model'}?`}
          description={`This downloads ${target?.label ?? 'the text model'}${target ? ` (about ${formatBytes(target.bytes)})` : ''} and runs it on this device through the bundled engine. Nothing is uploaded.`}
          footer={
            <>
              <DialogClose asChild><Button variant="ghost">Not now</Button></DialogClose>
              <Button
                variant="primary"
                disabled={!agreed}
                busy={installing}
                onClick={() => void install()}
              >
                Accept and download
              </Button>
            </>
          }
        >
          <label className="consent">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
            <span>
              I understand this downloads {target?.label ?? 'a text model'} to this device and runs it locally. Model weights follow their upstream Apache-2.0 terms.
            </span>
          </label>
        </DialogContent>
      </Dialog>
    </section>
  );
}

export function Models() {
  const { loading } = useStore();
  return (
    <div className="page narrow">
      <PageHeader title="Models" />
      {loading ? <Spinner label="Loading" /> : null}
      <Engine />
      <Compute />
      <Cleanup />
    </div>
  );
}
