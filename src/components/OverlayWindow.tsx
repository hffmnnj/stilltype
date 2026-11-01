import { useEffect, useState } from 'react';
import { api, errorMessage, isDesktop, listenNative } from '../lib/api';
import type { OverlayAction, OverlaySnapshot } from '../lib/types';
import { OverlayView } from './OverlayView';

/**
 * Native overlay entry (`index.html?overlay`). Renders outside every
 * provider: no store, no history, no camera. It mirrors the snapshot the
 * main window publishes and routes button presses back through
 * `overlay_action`, which the main window handles. In a browser the same
 * compact presentation renders the honest readiness snapshot directly.
 */
export function OverlayWindow() {
  const [snapshot, setSnapshot] = useState<OverlaySnapshot | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    (async () => {
      try {
        // Subscribe first so a snapshot published during startup is not lost.
        unlisten = await listenNative<OverlaySnapshot>('overlay-state', (s) => {
          if (!cancelled) setSnapshot(s);
        });
        if (cancelled) {
          unlisten();
          return;
        }
        setSnapshot(await api.getOverlayState());
      } catch (e) {
        if (!cancelled) setFatal(errorMessage(e));
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = snapshot?.theme ?? 'light';
    document.documentElement.style.colorScheme = snapshot?.theme ?? 'light';
  }, [snapshot?.theme]);

  const act = (action: OverlayAction) => {
    if (!isDesktop) {
      if (action === 'open') {
        window.location.assign(window.location.pathname);
        return;
      }
      if (action === 'dismiss') {
        setDismissed(true);
        return;
      }
      void api.overlayAction(action).catch((e) => setActionError(errorMessage(e)));
      return;
    }
    void api.overlayAction(action).catch(() => undefined);
  };

  useEffect(() => {
    // Best-effort local key handling only: the native window is
    // non-focusable to preserve the paste target, so this rarely fires.
    // Never a registered global shortcut.
    const on = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (document.querySelector('[role="dialog"]')) return;
      e.preventDefault();
      const phase = snapshot?.phase;
      act(phase === 'recording' || phase === 'opening' ? 'cancel' : 'dismiss');
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [snapshot?.phase]);

  if (dismissed && !isDesktop) {
    return <div className="overlay-root" />;
  }

  const viewSnapshot: OverlaySnapshot | null =
    snapshot && actionError && !isDesktop
      ? { ...snapshot, phase: 'error', message: actionError }
      : snapshot;

  return (
    <div className="overlay-root">
      {fatal ? (
        <div className="overlay-surface" data-phase="error" role="alert" aria-label={`Overlay unavailable: ${fatal}`}>
          <div className="overlay-row">
            <p className="overlay-status">
              <span>Overlay unavailable</span>
            </p>
          </div>
          <p className="overlay-message" title={fatal}>
            {fatal}
          </p>
        </div>
      ) : viewSnapshot ? (
        <OverlayView snapshot={viewSnapshot} onAction={act} />
      ) : (
        <div className="overlay-surface" data-phase="ready" role="status" aria-label="Loading overlay">
          <div className="overlay-row">
            <p className="overlay-status">
              <span>Starting</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
