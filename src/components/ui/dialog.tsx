import * as D from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import type { ReactNode } from 'react';

export const Dialog = D.Root;
export const DialogTrigger = D.Trigger;
export const DialogClose = D.Close;

export function DialogContent({
  title,
  description,
  children,
  footer,
  wide,
}: {
  title: string;
  description: string;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  return (
    <D.Portal>
      <D.Overlay className="dialog-overlay" />
      <D.Content className={wide ? 'dialog dialog-wide' : 'dialog'}>
        <div className="dialog-head">
          <D.Title className="dialog-title">{title}</D.Title>
          <D.Close className="icon-btn" aria-label="Close">
            <X size={16} aria-hidden />
          </D.Close>
        </div>
        <D.Description className="dialog-desc">{description}</D.Description>
        {children ? <div className="dialog-body">{children}</div> : null}
        {footer ? <div className="dialog-foot">{footer}</div> : null}
      </D.Content>
    </D.Portal>
  );
}
