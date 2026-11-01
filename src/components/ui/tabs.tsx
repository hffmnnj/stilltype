import type { ReactNode } from 'react';
import * as T from '@radix-ui/react-tabs';
export const Tabs = T.Root;
export const TabsList = ({ label, children }: { label: string; children: ReactNode }) => (
  <T.List className="tabs" aria-label={label}>{children}</T.List>
);
export const TabsTrigger = ({ value, children }: { value: string; children: ReactNode }) => (
  <T.Trigger className="tab" value={value}>{children}</T.Trigger>
);
export const TabsContent = ({ value, children }: { value: string; children: ReactNode }) => (
  <T.Content className="tab-panel" value={value}>{children}</T.Content>
);
