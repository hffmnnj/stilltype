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

/** The signature: a closed mouth, drawn as a single seam with soft corners. */
export function Seam({ size = 56 }: { size?: number }) {
  return (
    <svg className="seam" width={size} height={size * 0.5} viewBox="0 0 56 28" aria-hidden>
      <path d="M4 14 C 14 8, 22 11, 28 14 C 34 11, 42 8, 52 14 C 42 20, 34 19, 28 17.4 C 22 19, 14 20, 4 14 Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M4 14 C 18 15.5, 38 15.5, 52 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function Mark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className="mark">
      <rect width="32" height="32" rx="8" className="mark-bg" />
      <path d="M7 16 C 11 12, 14 13.5, 16 14.6 C 18 13.5, 21 12, 25 16 C 21 20, 18 18.6, 16 17.6 C 14 18.6, 11 20, 7 16 Z" fill="none" className="mark-lips" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M7 16 C 12 16.8, 20 16.8, 25 16" className="mark-seam" strokeWidth="2" strokeLinecap="round" fill="none" />
    </svg>
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
