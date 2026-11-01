import { AlertTriangle, CheckCircle2, Download, RefreshCw } from 'lucide-react';
import { useState } from 'react';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Button, Notice, PageHeader, Segmented, Spinner } from '../components/ui/misc';
import { Toggle } from '../components/ui/switch';
import { api, errorMessage, isDesktop } from '../lib/api';
import { useStore } from '../lib/store';
import type { Settings } from '../lib/types';

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

function Cleanup() {
  const { settings, updateSettings } = useStore();
  const [models, setModels] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    setBusy(true);
    setErr(null);
    try {
      setModels(await api.ollamaModels());
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const options = models && !models.includes(settings.cleanupModel) ? [settings.cleanupModel, ...models] : models;

  return (
    <section className="block" aria-labelledby="oll-h">
      <h2 id="oll-h" className="section-title">Transcript cleanup</h2>
      <Toggle
        checked={settings.cleanupEnabled}
        onCheckedChange={(v) => void updateSettings({ cleanupEnabled: v })}
        label="Tidy transcripts with Ollama"
        hint="Optional. Fixes casing and punctuation using a model on your own machine. The raw output is always kept."
      />
      <div className="row">
        <div className="row-text">
          <label htmlFor="oll-model" className="row-label">Model</label>
          <p className="row-hint">Only talks to Ollama at 127.0.0.1:11434 on this device.</p>
        </div>
        <div className="inline">
          {options && options.length > 0 ? (
            <select id="oll-model" className="field select" value={settings.cleanupModel} onChange={(e) => void updateSettings({ cleanupModel: e.target.value })}>
              {options.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          ) : (
            <input id="oll-model" className="field" value={settings.cleanupModel} readOnly aria-readonly />
          )}
          <Button onClick={() => void refresh()} busy={busy} disabled={!isDesktop}>
            {!busy ? <RefreshCw size={15} aria-hidden /> : null} Refresh
          </Button>
        </div>
      </div>
      {models && models.length === 0 ? (
        <Notice tone="warn">Ollama is running but has no models. Pull one with <code>ollama pull</code>.</Notice>
      ) : null}
      {err ? <Notice tone="error">{err}</Notice> : null}
      {!isDesktop ? <p className="muted small">Model listing is available in the desktop app.</p> : null}
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
