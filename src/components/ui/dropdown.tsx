import type { ReactNode } from 'react';
import * as M from '@radix-ui/react-dropdown-menu';
export const Menu = M.Root;
export const MenuTrigger = M.Trigger;
export const MenuContent = ({ children, label }: { children: ReactNode; label: string }) => (
  <M.Portal>
    <M.Content className="menu" align="end" sideOffset={6} aria-label={label}>{children}</M.Content>
  </M.Portal>
);
export const MenuItem = ({ children, onSelect, danger }: { children: ReactNode; onSelect: () => void; danger?: boolean }) => (
  <M.Item className={danger ? 'menu-item danger' : 'menu-item'} onSelect={onSelect}>{children}</M.Item>
);
