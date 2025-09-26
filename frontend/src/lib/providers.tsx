"use client";

import React from 'react';

type ProvidersProps = { children: React.ReactNode };

export function Providers({ children }: ProvidersProps) {
  return <>{children}</>;
}

// TEST: Providers renders children without error
