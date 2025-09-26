"use client";

import { useMemo } from 'react';

export function useAuth() {
  // Mock auth for Sprint 1
  const isAuthenticated = false;
  const user = isAuthenticated ? { id: '1', name: 'Admin' } : null;
  return useMemo(() => ({ isAuthenticated, user }), [isAuthenticated, user]);
}

// TEST: Hook returns stable object
