import * as D from '@radix-ui/react-dialog';
import { Command } from 'cmdk';
import { BookA, Camera, CameraOff, Circle, Cpu, History, Moon, ScanFace, Settings, Square, Sun, SunMoon } from 'lucide-react';
import type { ReactNode } from 'react';
import { useDictation } from '../lib/dictation';
import { useStore, type View } from '../lib/store';
import type { Theme } from '../lib/types';

export function CommandMenu() {
  const { commandOpen, setCommandOpen, go, updateSettings } = useStore();
  const d = useDictation();
  const run = (fn: () => void) => () => {
    setCommandOpen(false);
    fn();
  };
  const nav = (v: View, label: string, icon: ReactNode) => (
    <Command.Item value={`go ${label}`} onSelect={run(() => go(v))}>{icon}{label}</Command.Item>
  );
  const theme = (t: Theme, label: string, icon: ReactNode) => (
    <Command.Item value={`theme ${label}`} onSelect={run(() => void updateSettings({ theme: t }))}>{icon}{label}</Command.Item>
  );
  return (
    <D.Root open={commandOpen} onOpenChange={setCommandOpen}>
      <D.Portal>
        <D.Overlay className="dialog-overlay" />
        <D.Content className="dialog cmd-dialog" aria-describedby={undefined}>
          <D.Title className="visually-hidden">Command menu</D.Title>
          <Command label="Commands" className="cmd">
            <Command.Input placeholder="Type a command" className="cmd-input" />
            <Command.List className="cmd-list">
              <Command.Empty className="cmd-empty">No command matches.</Command.Empty>
              <Command.Group heading="Dictation">
                {d.phase === 'recording' ? (
                  <Command.Item value="stop dictating" onSelect={run(() => void d.stop())}><Square size={16} aria-hidden />Stop dictating</Command.Item>
                ) : (
                  <Command.Item value="start dictating" onSelect={run(() => void d.start())}><Circle size={16} aria-hidden />Start dictating</Command.Item>
                )}
                {d.cameraOpen ? (
                  <Command.Item value="close camera" onSelect={run(d.closeCamera)}><CameraOff size={16} aria-hidden />Close camera</Command.Item>
                ) : (
                  <Command.Item value="open camera" onSelect={run(() => { go('dictate'); void d.openCamera(); })}><Camera size={16} aria-hidden />Open camera</Command.Item>
                )}
              </Command.Group>
              <Command.Group heading="Go to">
                {nav('dictate', 'Dictate', <ScanFace size={16} aria-hidden />)}
                {nav('history', 'History', <History size={16} aria-hidden />)}
                {nav('vocabulary', 'Vocabulary', <BookA size={16} aria-hidden />)}
                {nav('models', 'Models', <Cpu size={16} aria-hidden />)}
                {nav('settings', 'Settings', <Settings size={16} aria-hidden />)}
              </Command.Group>
              <Command.Group heading="Theme">
                {theme('system', 'System theme', <SunMoon size={16} aria-hidden />)}
                {theme('light', 'Light theme', <Sun size={16} aria-hidden />)}
                {theme('dark', 'Dark theme', <Moon size={16} aria-hidden />)}
              </Command.Group>
            </Command.List>
          </Command>
        </D.Content>
      </D.Portal>
    </D.Root>
  );
}
