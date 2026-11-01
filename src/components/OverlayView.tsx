import { Copy, ExternalLink, Loader2, Play, Square, X } from 'lucide-react';
import type { ReactNode } from 'react';
import { formatClock } from '../lib/utils';
import type { OverlayAction, OverlayPhase, OverlaySnapshot } from '../lib/types';
import { Mark } from './ui/misc';

export type OverlayHandler = (action: OverlayAction) => void;

const STATUS: Record<OverlayPhase, string> = {
  ready: 'Ready',
  opening: 'Opening camera',
  recording: 'Dictating',
  processing: 'Transcribing',
  done: 'Done',
  error: 'Something went wrong',
};

interface RowButton {
  action: OverlayAction;
  label: string;
  icon: ReactNode;
  text?: string;
  kind: 'go' | 'stop' | 'open' | 'tool';
}

/**
 * Compact row actions. Opening has a single Cancel control (no duplicate);
 * processing keeps dismiss only; error pairs a clear Open with dismiss.
 */
function rowButtons(phase: OverlayPhase, hasText: boolean): RowButton[] {
  switch (phase) {
    case 'ready':
      return [{ action: 'start', label: 'Start dictating', icon: <Play size={14} aria-hidden />, kind: 'go' }];
    case 'opening':
      return [];
    case 'recording':
      return [{ action: 'stop', label: 'Stop and transcribe', icon: <Square size={12} fill="currentColor" aria-hidden />, text: 'Stop', kind: 'stop' }];
    case 'processing':
      return [];
    case 'done':
      return hasText
        ? [
            { action: 'copy', label: 'Copy result', icon: <Copy size={14} aria-hidden />, kind: 'tool' },
            { action: 'open', label: 'Open in main window', icon: <ExternalLink size={14} aria-hidden />, kind: 'tool' },
          ]
        : [];
    case 'error':
      return [{ action: 'open', label: 'Open main window', icon: <ExternalLink size={13} aria-hidden />, text: 'Open', kind: 'open' }];
  }
}

export function OverlayView({ snapshot, onAction }: { snapshot: OverlaySnapshot; onAction: OverlayHandler }) {
  const { phase, elapsed, text, message, cameraOpen } = snapshot;
  const buttons = rowButtons(phase, text.trim().length > 0);
  const busy = phase === 'opening' || phase === 'processing';
  const statusText = phase === 'ready' && message ? message : STATUS[phase];
  const showText = phase === 'done' && text.trim().length > 0;
  const showError = phase === 'error' && message;
  const showCameraNote = phase === 'recording' && !cameraOpen;
  const showTime = phase === 'opening' || phase === 'recording' || phase === 'processing';
  const dismissLabel =
    phase === 'recording' ? 'Cancel dictation' : phase === 'opening' ? 'Cancel' : 'Dismiss overlay';

  return (
    <div className="overlay-surface" data-phase={phase} role="status" aria-label={`Dictation overlay: ${STATUS[phase]}`}>
      <div className="overlay-row">
        <Mark size={20} />
        <p className="overlay-status" title={statusText}>
          {busy ? <Loader2 size={13} className="spin" aria-hidden /> : null}
          <span>{statusText}</span>
        </p>
        {showTime ? (
          <span className="overlay-time mono" aria-label={`Elapsed ${formatClock(elapsed)}`}>
            {formatClock(elapsed)}
          </span>
        ) : null}
        <span className="overlay-tools">
          {buttons.map((b) => (
            <button
              key={b.action}
              type="button"
              className={`overlay-tool overlay-tool-${b.kind}`}
              onClick={() => onAction(b.action)}
              aria-label={b.label}
            >
              {b.icon}
              {b.text ? <span>{b.text}</span> : null}
            </button>
          ))}
          <button type="button" className="overlay-tool overlay-tool-dismiss" onClick={() => onAction('dismiss')} aria-label={dismissLabel}>
            <X size={15} aria-hidden />
          </button>
        </span>
      </div>
      {showText ? (
        <p className="overlay-text" title={text} aria-label={`Dictated text: ${text}`}>
          {text}
        </p>
      ) : null}
      {showError ? (
        <p className="overlay-message" title={message}>
          {message}
        </p>
      ) : null}
      {showCameraNote ? <p className="overlay-message">Camera is closed. Open it in the main window.</p> : null}
    </div>
  );
}
