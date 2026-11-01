import { Loader2 } from 'lucide-react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '../../lib/utils';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

export function Button({
  variant = 'secondary',
  busy,
  children,
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; busy?: boolean }) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || busy}
      aria-busy={busy || undefined}
      className={cn('btn', `btn-${variant}`, className)}
    >
      {busy ? <Loader2 size={15} className="spin" aria-hidden /> : null}
      {children}
    </button>
  );
}

export function IconButton({
  label,
  children,
  pressed,
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; pressed?: boolean }) {
  return (
    <button type="button" {...rest} aria-label={label} title={label} aria-pressed={pressed} className={cn('icon-btn', className)}>
      {children}
    </button>
  );
}

export function Keycaps({ keys }: { keys: string[] }) {
  return (
    <span className="keys">
      {keys.map((k, i) => (
        <kbd key={`${k}-${i}`} className="key">{k}</kbd>
      ))}
    </span>
  );
}

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1 className="page-title">{title}</h1>
      {children ? <div className="page-head-extra">{children}</div> : null}
    </header>
  );
}

export function Notice({
  tone = 'neutral',
  children,
  action,
}: {
  tone?: 'neutral' | 'error' | 'warn';
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={`notice notice-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="notice-text">{children}</div>
      {action}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <Seam />
      <h2 className="empty-title">{title}</h2>
      {children ? <p className="empty-text">{children}</p> : null}
      {action}
    </div>
  );
}

/** The signature: generated warm vermilion mouth mark on near-black (raster). Decorative. */
export function Seam({ size = 56 }: { size?: number }) {
  return (
    <img src="/brand/stilltype-mark.png" width={size} height={size} alt="" aria-hidden className="seam" />
  );
}

export function Mark({ size = 28 }: { size?: number }) {
  return (
    <img src="/brand/stilltype-mark.png" width={size} height={size} alt="" aria-hidden className="mark" />
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="spinner-row" role="status">
      <Loader2 size={16} className="spin" aria-hidden /> {label}
    </span>
  );
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset className="segmented" disabled={disabled}>
      <legend className="visually-hidden">{label}</legend>
      {options.map((o) => (
        <label key={o.value} className="seg" data-on={o.value === value || undefined}>
          <input type="radio" name={label} value={o.value} checked={o.value === value} onChange={() => onChange(o.value)} />
          <span>{o.label}</span>
        </label>
      ))}
    </fieldset>
  );
}
