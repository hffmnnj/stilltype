import * as S from '@radix-ui/react-switch';
import { useId, type ReactNode } from 'react';

export function Toggle({
  checked,
  onCheckedChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
  label: string;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="row">
      <div className="row-text">
        <label htmlFor={id} className="row-label">
          {label}
        </label>
        {hint ? <p className="row-hint" id={`${id}-hint`}>{hint}</p> : null}
      </div>
      <div className="switch-wrap">
        <span className="switch-state" aria-hidden>{checked ? 'On' : 'Off'}</span>
        <S.Root
          id={id}
          className="switch"
          checked={checked}
          onCheckedChange={onCheckedChange}
          disabled={disabled}
          aria-describedby={hint ? `${id}-hint` : undefined}
        >
          <S.Thumb className="switch-thumb" />
        </S.Root>
      </div>
    </div>
  );
}
