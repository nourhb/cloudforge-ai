"use client";

import * as React from 'react';
import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { WorkerUpload } from '@/components/WorkerUpload';

type WorkerMeta = {
  name: string;
  runtime: string;
  filePath: string;
  uploadedAt: string;
};

export default function MarketplacePage() {
  const [items, setItems] = useState<WorkerMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchList() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'}/api/marketplace/list`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'Failed to load');
      setItems(data.items || []);
    } catch (e: any) {
      setError(e.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchList();
  }, []);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <h1 className="text-3xl font-bold">API Marketplace</h1>

      <section className="rounded-xl border p-4">
        <h2 className="text-xl font-semibold mb-3">Upload Worker</h2>
        <WorkerUpload />
        <div className="mt-3">
          <button className="text-sm underline" onClick={fetchList}>Refresh List</button>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Available Workers</h2>
        {loading && <div className="text-sm text-gray-600">Loading...</div>}
        {error && <div className="text-sm text-red-600">{error}</div>}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {items.map((w) => (
            <Card key={`${w.name}-${w.uploadedAt}`}>
              <CardHeader>
                <CardTitle>{w.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-2">Runtime: {w.runtime}</p>
                <p className="text-xs text-gray-500">Uploaded: {new Date(w.uploadedAt).toLocaleString()}</p>
              </CardContent>
            </Card>
          ))}
          {items.length === 0 && !loading && !error && (
            <div className="text-sm text-gray-600">No workers uploaded yet.</div>
          )}
        </div>
      </section>
    </div>
  );
}

// TEST: Page loads, can fetch list, and renders upload form
