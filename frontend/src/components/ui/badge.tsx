import * as React from 'react';
import { cn } from '@/lib/utils';

type Variant = 'default' | 'secondary' | 'outline';

export function Badge({ className, variant = 'default', ...props }: React.HTMLAttributes<HTMLDivElement> & { variant?: Variant }) {
  const base = 'inline-flex items-center rounded-full px-3 py-1 text-xs font-medium';
  const variants: Record<Variant, string> = {
    default: 'bg-blue-600 text-white',
    secondary: 'bg-gray-200 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
    outline: 'border border-gray-300 text-gray-700 dark:border-gray-700 dark:text-gray-200',
  };
  return <div className={cn(base, variants[variant], className)} {...props} />;
}

// TEST: Badge renders
