import { Keyboard, RefreshCw, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useId, useState } from 'react';
import { toast } from 'sonner';
import { Dialog, DialogClose, DialogContent } from '../components/ui/dialog';
import { Button, Keycaps, Notice, PageHeader, Segmented } from '../components/ui/misc';
import { Toggle } from '../components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { OverlayView } from '../components/OverlayView';
import { api, errorMessage, isDesktop } from '../lib/api';
import { useDictation } from '../lib/dictation';
import { useStore } from '../lib/store';
import type { OverlayAction, Settings as S, ShortcutStatus } from '../lib/types';
import { acceleratorFromEvent, isLinux, modKey, shortcutKeys } from '../lib/utils';

function RangeRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  unit,
  format,
  onCommit,
  disabled,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  format?: (v: number) => string;
  onCommit: (v: number) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  const commit = () => {
    if (local !== value) onCommit(local);
  };
  return (
    <div className="row">
      <div className="row-text">
        <label htmlFor={id} className="row-label">{label}</label>
        {hint ? <p className="row-hint">{hint}</p> : null}
      </div>
      <div className="range-wrap">
        <input
          id={id}
          type="range"
          className="range"
          min={min}
          max={max}
          step={step}
          value={local}
          disabled={disabled}
          onChange={(e) => setLocal(Number(e.target.value))}
          onPointerUp={commit}
          onKeyUp={commit}
          onBlur={commit}
        />
        <output className="mono range-out" htmlFor={id}>{format ? format(local) : `${Number.isInteger(step) ? local : local.toFixed(1)}${unit ?? ''}`}</output>
      </div>
    </div>
  );
}

function ShortcutRow() {
  const { settings, updateSettings } = useStore();
  const [recording, setRecording] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="row">
      <div className="row-text">
        <p className="row-label">Global shortcut</p>
        <p className="row-hint">
          {isDesktop ? 'Works from any app.' : 'System-wide shortcuts need the desktop app.'}
          {isLinux ? ' On Wayland, the compositor may block global shortcuts; if it does, dictate from this window.' : ''}
        </p>
        {err ? <p className="field-error" role="alert">{err}</p> : null}
      </div>
      <button
        type="button"
        className="shortcut-btn"
        disabled={!isDesktop}
        data-recording={recording || undefined}
        onClick={() => { setErr(null); setRecording((r) => !r); }}
        onBlur={() => setRecording(false)}
        onKeyDown={(e) => {
          if (!recording) return;
          e.preventDefault();
          e.stopPropagation();
          if (e.key === 'Escape') { setRecording(false); return; }
          const accel = acceleratorFromEvent(e.nativeEvent);
          if (!accel) return;
          if (!accel.includes('+')) { setErr('Include at least one modifier key, such as Ctrl or Shift.'); return; }
          setRecording(false);
          void updateSettings({ shortcut: accel }).then((ok) => { if (!ok) setErr('The system refused that shortcut. It may be taken by another app.'); });
        }}
        aria-label={recording ? 'Press the new shortcut, or Escape to cancel' : `Change shortcut, currently ${shortcutKeys(settings.shortcut).join(' ')}`}
      >
        {recording ? <span className="muted">Press keys…</span> : <><Keyboard size={15} aria-hidden /> <Keycaps keys={shortcutKeys(settings.shortcut)} /></>}
      </button>
    </div>
  );
}

function FnRow() {
  const [status, setStatus] = useState<ShortcutStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const load = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      setStatus(await api.shortcutStatus());
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  const hint = !isDesktop
    ? 'System shortcuts need the desktop app.'
    : modKey === '⌘'
      ? 'The Fn key is hold-to-dictate and needs Input Monitoring permission. Grant it in System Settings, then restart Stilltype.'
      : 'Most PC firmware reserves the Fn key, so it is not a reliable trigger. Use the global shortcut below instead.';
  return (
    <div className="row">
      <div className="row-text">
        <p className="row-label">Fn key</p>
        <p className="row-hint">{hint}</p>
        <p className="mono small" role="status">{busy ? 'Checking…' : (err ?? status?.message ?? '')}</p>
      </div>
      <div className="inline">
        <Button onClick={() => void load()} busy={busy} disabled={busy}>
          {!busy ? <RefreshCw size={15} aria-hidden /> : null} Refresh
        </Button>
      </div>
    </div>
  );
}

function OverlayRow() {
  const d = useDictation();
  const { go } = useStore();
  const [preview, setPreview] = useState(false);
  const onAction = (action: OverlayAction) => {
    if (action === 'dismiss') setPreview(false);
    else if (action === 'copy') void d.copyDraft();
    else if (action === 'open') {
      setPreview(false);
      go('dictate');
    } else if (action === 'preview') d.previewOverlay();
    else if (action === 'start') void d.start();
    else if (action === 'stop') void d.stop();
    else if (action === 'cancel') void d.cancel();
  };
  return (
    <div className="row row-stack">
      <div className="row-text">
        <p className="row-label">Overlay</p>
        <p className="row-hint">
          A compact window that follows dictation from any app. The preview below shows live readiness in this
          page without starting the camera{isDesktop ? '; “Show in overlay window” opens the native window.' : '.'}
        </p>
      </div>
      <div className="inline">
        <Button onClick={() => setPreview((v) => !v)} aria-expanded={preview}>
          {preview ? 'Hide preview' : 'Preview overlay'}
        </Button>
        {isDesktop ? <Button onClick={() => d.previewOverlay()}>Show in overlay window</Button> : null}
      </div>
      {preview ? (
        <div className="overlay-preview">
          <OverlayView snapshot={d.overlaySnapshot()} onAction={onAction} />
        </div>
      ) : null}
    </div>
  );
}

function ContextRow() {
  const { settings, updateSettings } = useStore();
  const id = useId();
  const [local, setLocal] = useState(settings.context);
  useEffect(() => setLocal(settings.context), [settings.context]);
  return (
    <div className="row row-stack">
      <div className="row-text">
        <label htmlFor={id} className="row-label">Context for intent correction</label>
        <p className="row-hint">
          Optional. A line about what you usually write, used only in Intent mode.
          {settings.cleanupMode === 'intent' ? '' : ' Intent mode is off, so it is saved but not used.'}
        </p>
      </div>
      <textarea
        id={id}
        className="field"
        rows={2}
        maxLength={400}
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={() => { if (local !== settings.context) void updateSettings({ context: local }); }}
      />
    </div>
  );
}

export function SettingsPage() {
  const { settings, updateSettings, settingsError, settingsSaving, clearHistory, history } = useStore();
  const d = useDictation();
  const camId = useId();
  const retId = useId();
  const [confirm, setConfirm] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearErr, setClearErr] = useState<string | null>(null);
  const set = (patch: Partial<S>) => void updateSettings(patch);

  const retentionOptions = [7, 30, 90, 365];
  if (!retentionOptions.includes(settings.retentionDays)) retentionOptions.push(settings.retentionDays);
  retentionOptions.sort((a, b) => a - b);

  const doClear = async () => {
    setClearing(true);
    setClearErr(null);
    try {
      await clearHistory();
      setConfirm(false);
      toast.success('History cleared');
    } catch (e) {
      setClearErr(errorMessage(e));
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="page narrow">
      <PageHeader title="Settings">
        <span className="save-state mono muted" role="status">{settingsSaving ? 'Saving…' : settingsError ? 'Not saved' : 'Saved automatically'}</span>
      </PageHeader>
      {settingsError ? <Notice tone="error">{settingsError}</Notice> : null}

      <Tabs defaultValue="camera">
        <TabsList label="Settings groups">
          <TabsTrigger value="camera">Camera</TabsTrigger>
          <TabsTrigger value="dictation">Dictation</TabsTrigger>
          <TabsTrigger value="appearance">Appearance</TabsTrigger>
          <TabsTrigger value="privacy">Privacy</TabsTrigger>
        </TabsList>

        <TabsContent value="camera">
          <div className="row">
            <div className="row-text">
              <label htmlFor={camId} className="row-label">Camera</label>
              <p className="row-hint">Names appear once you have opened the camera on Dictate.</p>
            </div>
            <div className="inline">
              <select id={camId} className="field select" value={settings.cameraId} onChange={(e) => set({ cameraId: e.target.value })}>
                <option value="">System default</option>
                {settings.cameraId && !d.devices.some((x) => x.deviceId === settings.cameraId) ? (
                  <option value={settings.cameraId}>Saved camera (not found)</option>
                ) : null}
                {d.devices.filter((x) => x.kind === 'videoinput').map((x, i) => (
                  <option key={x.deviceId || i} value={x.deviceId}>{x.label || `Camera ${i + 1}`}</option>
                ))}
              </select>
              <Button onClick={() => void d.refreshDevices()}><RefreshCw size={15} aria-hidden /> Refresh</Button>
            </div>
          </div>
          <Toggle checked={settings.showPreview} onCheckedChange={(v) => set({ showPreview: v })} label="Show preview" hint="Hiding the preview does not turn the camera off." />
          <Toggle checked={settings.mirrored} onCheckedChange={(v) => set({ mirrored: v })} label="Mirror preview" hint="Only changes what you see. Frames are read unmirrored." />
        </TabsContent>

        <TabsContent value="dictation">
          <FnRow />
          <ShortcutRow />
          <div className="row">
            <div className="row-text">
              <p className="row-label">Mode</p>
              <p className="row-hint">{settings.dictationMode === 'hold' ? 'Hold the shortcut while you mouth, release to transcribe.' : 'Press to start. Pauses split the clip; press again to finish.'}</p>
            </div>
            <Segmented<S['dictationMode']>
              label="Dictation mode"
              value={settings.dictationMode}
              onChange={(v) => set({ dictationMode: v })}
              options={[{ value: 'toggle', label: 'Toggle' }, { value: 'hold', label: 'Hold' }]}
            />
          </div>
          <Toggle
            checked={settings.autoPaste}
            onCheckedChange={(v) => set({ autoPaste: v })}
            disabled={!isDesktop}
            label="Paste into the active app"
            hint={
              isDesktop
                ? modKey === '⌘'
                  ? 'Needs Accessibility permission in System Settings to send the paste keystroke.'
                  : isLinux
                    ? 'On Wayland, apps may refuse synthetic paste. The text is still copied for you to paste.'
                    : 'Sends a paste keystroke after each transcript.'
                : 'Desktop app only.'
            }
          />
          <ContextRow />
          <OverlayRow />
          <RangeRow label="Beam size" hint="5 is a good balance. Larger beams use more memory and may be slower." value={settings.beamSize} min={1} max={60} step={1} onCommit={(v) => set({ beamSize: v })} />
          <RangeRow label="Pause length" hint="Silence that ends a phrase in toggle mode." value={settings.pauseSeconds} min={0.3} max={2} step={0.1} unit="s" onCommit={(v) => set({ pauseSeconds: v })} disabled={settings.dictationMode === 'hold'} />
          <RangeRow label="Recording limit" hint="Recording stops automatically at this length. Two minutes is the longest." value={settings.maxClipSeconds} min={5} max={120} step={5} format={(v) => (v >= 60 ? `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}` : `${v}s`)} onCommit={(v) => set({ maxClipSeconds: v })} />
        </TabsContent>

        <TabsContent value="appearance">
          <div className="row">
            <div className="row-text">
              <p className="row-label">Theme</p>
              <p className="row-hint">System follows your operating system.</p>
            </div>
            <Segmented<S['theme']>
              label="Theme"
              value={settings.theme}
              onChange={(v) => set({ theme: v })}
              options={[{ value: 'system', label: 'System' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]}
            />
          </div>
        </TabsContent>

        <TabsContent value="privacy">
          <Toggle checked={settings.keepHistory} onCheckedChange={(v) => set({ keepHistory: v })} label="Keep history" hint="Stored in plain files on this device. Stilltype does not encrypt them." />
          <div className="row">
            <div className="row-text">
              <label htmlFor={retId} className="row-label">Delete entries after</label>
              <p className="row-hint">Older entries are removed automatically.</p>
            </div>
            <select id={retId} className="field select" value={settings.retentionDays} disabled={!settings.keepHistory} onChange={(e) => set({ retentionDays: Number(e.target.value) })}>
              {retentionOptions.map((n) => <option key={n} value={n}>{n === 365 ? '1 year' : `${n} days`}</option>)}
            </select>
          </div>
          <div className="row">
            <div className="row-text">
              <p className="row-label">Clear history</p>
              <p className="row-hint">{history.length} {history.length === 1 ? 'entry' : 'entries'} on this device.</p>
            </div>
            <Button variant="danger" disabled={history.length === 0} onClick={() => setConfirm(true)}><Trash2 size={15} aria-hidden /> Clear history</Button>
          </div>
          <p className="muted small pad-top">Camera frames are processed on this device and are not uploaded. Stilltype records no audio.</p>
        </TabsContent>
      </Tabs>

      <Dialog open={confirm} onOpenChange={(o) => { setConfirm(o); if (!o) setClearErr(null); }}>
        <DialogContent
          title="Clear all history?"
          description="This permanently removes every entry from this device. Vocabulary is kept."
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
