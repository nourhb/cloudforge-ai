import * as React from 'react';

export function Progress({ value = 0, className }: { value?: number; className?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div className={"progress-bar "+(className||'')} aria-valuenow={v} aria-valuemin={0} aria-valuemax={100} role="progressbar">
      <div className="progress-fill" style={{ width: `${v}%` }} />
    </div>
  );
}

// TEST: Progress shows width
