import * as React from 'react';

export function Tabs({ value, onValueChange, children, className }: any) {
  return <div className={className}>{React.Children.map(children, (child: any) => React.cloneElement(child, { value, onValueChange }))}</div>;
}

export function TabsList({ children, className }: any) {
  return <div className={`grid grid-cols-2 gap-2 rounded-lg border p-1 ${className || ''}`}>{children}</div>;
}

export function TabsTrigger({ value, children, onValueChange }: any) {
  const active = (parentValue: string) => parentValue === value;
  return (
    <button
      className="rounded-md px-4 py-2 text-sm font-medium data-[state=active]:bg-blue-600 data-[state=active]:text-white"
      data-state={active((onValueChange as any)?.name) ? 'active' : undefined}
      onClick={() => onValueChange?.(value)}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children }: any) {
  // Always render for simplicity; real impl would context-provide value
  return <div>{children}</div>;
}

// TEST: Minimal tabs for UI
