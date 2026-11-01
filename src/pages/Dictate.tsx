import { Camera, CameraOff, Check, Circle, Clipboard, ClipboardPaste, Eye, EyeOff, Save, Square, X } from 'lucide-react';
import { useId } from 'react';
import { Button, IconButton, Keycaps, Notice, PageHeader, Seam, Spinner } from '../components/ui/misc';
import { useDictation } from '../lib/dictation';
import { isDesktop } from '../lib/api';
import { useStore } from '../lib/store';
import { formatClock, modKey, shortcutKeys } from '../lib/utils';

function Guide({ active }: { active: boolean }) {
  return (
    <svg className="guide" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden focusable="false">
      <g className="guide-corners">
        <path d="M18 46V18h28M354 18h28v28M382 254v28h-28M46 282H18v-28" />
      </g>
      {active ? (
        <>
          <ellipse className="guide-face" cx="200" cy="146" rx="84" ry="110" />
          <line className="guide-eye" x1="112" x2="288" y1="116" y2="116" />
          <rect className="guide-mouth" x="158" y="188" width="84" height="38" rx="8" />
          <text className="guide-text" x="294" y="119">eye line</text>
          <text className="guide-text" x="248" y="212">mouth crop</text>
        </>
      ) : null}
    </svg>
  );
}

function Stage() {
  const d = useDictation();
  const { settings } = useStore();
  const showVideo = d.cameraOpen && settings.showPreview;
  const recording = d.phase === 'recording';
  return (
    <div className="stage" data-recording={recording || undefined} data-live={d.cameraOpen || undefined}>
      <video
        ref={d.videoRef}
        className="stage-video"
        muted
        playsInline
        aria-label="Camera preview"
        style={{ transform: settings.mirrored ? 'scaleX(-1)' : undefined, opacity: showVideo ? 1 : 0 }}
      />
      <Guide active={d.cameraOpen} />
      {!d.cameraOpen ? (
        <div className="stage-empty">
          <Seam size={84} />
          <p className="stage-line">Say it without a sound.</p>
          <p className="stage-sub">Your camera stays off until you open it. Nothing leaves this device.</p>
          {d.phase === 'opening' ? (
            <Spinner label="Waiting for camera…" />
          ) : (
            <Button variant="primary" onClick={() => void d.openCamera()}>
              <Camera size={16} aria-hidden /> Open camera
            </Button>
          )}
        </div>
      ) : null}
      {d.cameraOpen && !settings.showPreview ? (
        <div className="stage-empty stage-quiet">
          <EyeOff size={20} aria-hidden />
          <p className="stage-sub">Preview hidden. The camera is still on{recording ? ' and recording' : ''}.</p>
        </div>
      ) : null}
      {recording ? (
        <div className="stage-badge" role="status">
          <span className="rec-dot" aria-hidden />
          <span>Recording</span>
          <span className="mono">
            {formatClock(d.elapsed)} / {formatClock(settings.maxClipSeconds)}
          </span>
        </div>
      ) : null}
      {d.cameraOpen && d.quality ? (
        <div className="stage-quality" aria-live="polite">
          <span className={d.quality.faceDetected ? 'q-ok' : 'q-warn'}>
            {d.quality.faceDetected ? 'Face found' : 'No face'}
          </span>
          <span className="mono">{Math.round(d.quality.fps)} fps</span>
          {d.quality.hint ? <span className="q-hint">{d.quality.hint}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function Controls() {
  const d = useDictation();
  const { settings, engine, go } = useStore();
  const recording = d.phase === 'recording';
  const processing = d.phase === 'processing';
  const hintId = useId();
  const keys = isDesktop ? shortcutKeys(settings.shortcut) : [modKey, '↵'];
  const needsSetup = isDesktop && !engine?.ready;
  const label = recording ? 'Stop' : processing ? 'Transcribing' : 'Start dictating';
  return (
    <div className="controls">
      <div className="controls-main">
        <button
          type="button"
          className="record"
          data-on={recording || undefined}
          onClick={() => void d.toggle()}
          disabled={processing || d.phase === 'opening' || (!recording && Boolean(d.blockedReason) && !isDesktop)}
          aria-label={label}
          aria-describedby={d.blockedReason ? hintId : undefined}
        >
          {recording ? <Square size={20} fill="currentColor" aria-hidden /> : <Circle size={22} fill="currentColor" aria-hidden />}
        </button>
        <div className="controls-copy">
          <p className="controls-title">{label}</p>
          {recording ? (
            <p className="controls-sub">
              <Keycaps keys={['Esc']} /> cancels without transcribing
            </p>
          ) : (
            <p className="controls-sub">
              <Keycaps keys={keys} /> {isDesktop ? (settings.dictationMode === 'hold' ? 'hold to dictate' : 'press to start and stop') : 'starts and stops'}
            </p>
          )}
        </div>
      </div>
      <div className="controls-side">
        {d.cameraOpen ? (
          <Button onClick={d.closeCamera} disabled={recording || processing}>
            <CameraOff size={16} aria-hidden /> Close camera
          </Button>
        ) : null}
      </div>
      {d.blockedReason && !recording ? (
        <p className="controls-reason" id={hintId}>
          {d.blockedReason}{' '}
          {needsSetup ? (
            <button type="button" className="link" onClick={() => go('models')}>
              Go to Models
            </button>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

function Setup() {
  const d = useDictation();
  const { engine, settings, go, loading } = useStore();
  const camOk = d.cameraOpen;
  const engOk = Boolean(engine?.ready);
  const items = [
    {
      key: 'camera',
      done: camOk,
      title: 'Camera',
      body: camOk ? 'On. Face the lens with your mouth in view.' : 'Off until you open it.',
      action: camOk ? null : (
        <Button onClick={() => void d.openCamera()} disabled={d.phase === 'opening'}>Open</Button>
      ),
    },
    {
      key: 'engine',
      done: engOk,
      title: 'Research engine',
      body: engOk
        ? `USR 2.0 Base+ ready on ${engine?.device ?? 'this device'}.`
        : isDesktop
          ? 'USR 2.0 Base+ (2026), about 1.4 GB, runs on your GPU or CPU.'
          : 'Runs in the desktop app only.',
      action: !engOk && isDesktop ? <Button variant={camOk ? 'primary' : 'secondary'} onClick={() => go('models')}>Set up</Button> : null,
    },
    {
      key: 'shortcut',
      done: isDesktop,
      title: 'Shortcut',
      body: isDesktop
        ? settings.dictationMode === 'hold'
          ? 'Hold to dictate from any app.'
          : 'Press to start, press again to stop.'
        : 'System-wide shortcuts need the desktop app.',
      extra: isDesktop ? <Keycaps keys={shortcutKeys(settings.shortcut)} /> : null,
      action: isDesktop ? <Button variant="ghost" onClick={() => go('settings')}>Change</Button> : null,
    },
  ];
  const done = items.filter((i) => i.done).length;
  return (
    <section className="aside-block" aria-labelledby="setup-h">
      <div className="aside-head">
        <h2 id="setup-h" className="aside-title">{done === items.length ? 'Ready' : 'Setup'}</h2>
        <span className="mono muted">{loading ? '…' : `${done} of ${items.length}`}</span>
      </div>
      <ol className="checklist">
        {items.map((i) => (
          <li key={i.key} className="check" data-done={i.done || undefined}>
            <span className="check-mark" aria-hidden>{i.done ? <Check size={13} strokeWidth={3} /> : null}</span>
            <div className="check-body">
              <p className="check-title">
                {i.title}
                <span className="visually-hidden">{i.done ? ', done' : ', not done'}</span>
              </p>
              <p className="check-text">{i.body}</p>
              {'extra' in i && i.extra ? i.extra : null}
            </div>
            {i.action}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Tips() {
  return (
    <section className="aside-block" aria-labelledby="tips-h">
      <h2 id="tips-h" className="aside-title">For a clean read</h2>
      <ul className="tips">
        <li>Light your face from the front, not from behind.</li>
        <li>Keep your head still and your lips in the crop box.</li>
        <li>Mouth each word a little slower and wider than you would in speech.</li>
      </ul>
    </section>
  );
}

function Transcript() {
  const d = useDictation();
  const { settings } = useStore();
  const draft = d.draft;
  const rawDiffers = draft && draft.raw.trim() !== '' && draft.raw.trim() !== draft.text.trim() && !draft.dirty;
  return (
    <section className="transcript" aria-labelledby="tr-h">
      <div className="transcript-head">
        <h2 id="tr-h" className="section-title">Latest transcript</h2>
        {draft ? (
          <span className="mono muted">
            {(draft.durationMs / 1000).toFixed(1)}s clip · {(draft.inferenceMs / 1000).toFixed(1)}s to read
          </span>
        ) : null}
      </div>
      {draft ? (
        <>
          <textarea
            className="transcript-text"
            value={draft.text}
            onChange={(e) => d.editDraft(e.target.value)}
            aria-labelledby="tr-h"
            rows={3}
            spellCheck
          />
          {rawDiffers ? (
            <details className="raw">
              <summary>Raw model output</summary>
              <p>{draft.raw}</p>
            </details>
          ) : null}
          <div className="transcript-actions">
            <Button onClick={() => void d.copyDraft()} disabled={!draft.text}>
              <Clipboard size={15} aria-hidden /> Copy
            </Button>
            {isDesktop ? (
              <Button onClick={() => void d.pasteDraft()} disabled={!draft.text}>
                <ClipboardPaste size={15} aria-hidden /> Paste into app
              </Button>
            ) : null}
            {draft.entryId ? (
              <Button onClick={() => void d.saveDraft()} disabled={!draft.dirty}>
                <Save size={15} aria-hidden /> Save edit
              </Button>
            ) : (
              <span className="muted small">
                {settings.keepHistory ? 'Not saved to history' : 'History is off'}
              </span>
            )}
            <IconButton label="Clear transcript" onClick={d.discardDraft} className="push-right">
              <X size={16} aria-hidden />
            </IconButton>
          </div>
        </>
      ) : (
        <p className="transcript-empty">Nothing yet. What you mouth appears here, ready to edit.</p>
      )}
    </section>
  );
}

export function Dictate() {
  const d = useDictation();
  const { settings, updateSettings } = useStore();
  return (
    <div className="page dictate">
      <PageHeader title="Dictate">
        {d.cameraOpen ? (
          <IconButton
            label={settings.showPreview ? 'Hide preview' : 'Show preview'}
            pressed={!settings.showPreview}
            onClick={() => void updateSettings({ showPreview: !settings.showPreview })}
          >
            {settings.showPreview ? <Eye size={17} aria-hidden /> : <EyeOff size={17} aria-hidden />}
          </IconButton>
        ) : null}
      </PageHeader>
      <div className="dictate-grid">
        <div className="dictate-main">
          <Stage />
          <Controls />
          {d.error ? (
            <Notice
              tone="error"
              action={
                <span className="notice-actions">
                  {!d.cameraOpen ? <Button onClick={() => void d.openCamera()}>Retry</Button> : null}
                  <Button variant="ghost" onClick={d.clearError}>Dismiss</Button>
                </span>
              }
            >
              {d.error}
            </Notice>
          ) : null}
          <Transcript />
        </div>
        <aside className="dictate-aside" aria-label="Setup and tips">
          <Setup />
          <Tips />
        </aside>
      </div>
    </div>
  );
}
