import { AlertTriangle, CheckCircle2, Download, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Button, Notice, PageHeader, Segmented, Spinner } from '../components/ui/misc';
import { Toggle } from '../components/ui/switch';
import { api, errorMessage, isDesktop, listenNative } from '../lib/api';
import type { InstallProgress } from '../lib/api';
import { useStore } from '../lib/store';
import type { CleanupStatus, Settings } from '../lib/types';

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

function Cleanup() {
  const { settings, updateSettings } = useStore();
  const [status, setStatus] = useState<CleanupStatus | null>(null);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState<InstallProgress | null>(null);
  const [consent, setConsent] = useState(false);
  const [agreed, setAgreed] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      setStatus(await api.cleanupStatus());
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

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
      setStatus(await api.installCleanup(true));
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setInstalling(false);
      setConsent(false);
    }
  };

  const ready = Boolean(status?.ready);
  const pct = Math.round((progress?.progress ?? 0) * 100);

  return (
    <section className="block" aria-labelledby="cleanup-h">
      <div className="block-head">
        <h2 id="cleanup-h" className="section-title">Transcript cleanup</h2>
        <span className={`status ${ready ? 'status-ok' : 'status-idle'}`}>{busy || !status ? 'Checking' : ready ? 'Ready' : 'Not installed'}</span>
      </div>
      <p className="lede">
        Tidies casing and punctuation on this device with a small bundled text model. The raw output is always kept.
      </p>

      {!isDesktop ? (
        <Notice tone="warn">
          Browser preview. The cleanup model installs and runs in the desktop app only.
        </Notice>
      ) : null}

      <dl className="kvs">
        <Row label="Engine" value={status ? (status.runtimeBundled ? 'Bundled (llama.cpp)' : 'Not bundled') : '…'} ok={status ? status.runtimeBundled : undefined} />
        <Row label="Cleanup model" value={status?.modelId ?? '…'} />
        <Row label="Download size" value={status ? formatBytes(status.totalBytes) : '…'} />
        <Row label="Compute" value={status?.device ?? '…'} />
      </dl>

      {status?.error ? <Notice tone="error">{status.error}</Notice> : null}
      {err ? (
        <Notice tone="error" action={<Button onClick={() => void refresh()}>Retry</Button>}>{err}</Notice>
      ) : null}

      {installing || (progress && progress.stage !== 'ready' && progress.stage !== 'error' && !ready) ? (
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
        disabled={!ready}
        label="Tidy transcripts after dictation"
        hint={ready ? 'Fixes casing and punctuation locally after each transcript.' : 'Download the model below to turn this on.'}
      />

      {!ready ? (
        <div className="actions">
          <Button variant="primary" disabled={!isDesktop || busy} busy={installing} onClick={() => { setAgreed(false); setConsent(true); }}>
            <Download size={16} aria-hidden /> {installing ? 'Downloading…' : 'Download cleanup model'}
          </Button>
          {status ? <span className="muted small">About {formatBytes(status.totalBytes)}.</span> : null}
        </div>
      ) : (
        <div className="actions">
          <Button onClick={() => void refresh()} busy={busy}>
            {!busy ? <RefreshCw size={15} aria-hidden /> : null} Check again
          </Button>
        </div>
      )}

      <Dialog open={consent} onOpenChange={setConsent}>
        <DialogContent
          title="Download the cleanup model?"
          description={`This downloads the text cleanup model${status ? ` (about ${formatBytes(status.totalBytes)})` : ''} and runs it on this device through the bundled engine. Nothing is uploaded.`}
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
              I understand this downloads a text cleanup model to this device and runs it locally. Model weights follow their upstream Apache-2.0 terms.
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
