"use client";

import { useEffect, useState } from 'react';

export function useSystemHealth() {
  const [health, setHealth] = useState<{ status: string } | null>(null);
  const [metrics, setMetrics] = useState<any>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let aborted = false;
    async function load() {
      setIsLoading(true);
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'}/health`);
        const data = await res.json().catch(() => ({ status: 'unknown' }));
        if (!aborted) setHealth(data);
      } catch (e) {
        if (!aborted) setHealth({ status: 'degraded' });
      } finally {
        if (!aborted) setIsLoading(false);
      }
    }
    load();
    return () => { aborted = true; };
  }, []);

  return { health, metrics, isLoading };
}

// TEST: Hook fetches /health from backend
