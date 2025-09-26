"use client";

import * as React from 'react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

export default function IacPage() {
  const [prompt, setPrompt] = useState('Expose backend as ClusterIP on port 4000');
  const [yaml, setYaml] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setYaml('');
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'}/api/iac/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (!res.ok || !data?.ok) throw new Error(data?.error || 'generation failed');
      setYaml(data.yaml || '');
    } catch (e: any) {
      setError(e.message || 'generation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-4">
      <h1 className="text-3xl font-bold">AI IaC Generator</h1>
      <form onSubmit={onGenerate} className="space-y-3">
        <textarea className="w-full min-h-[120px] rounded border p-2" value={prompt} onChange={e => setPrompt(e.target.value)} />
        <Button type="submit" disabled={loading}>{loading ? 'Generating...' : 'Generate YAML'}</Button>
      </form>
      {error && <div className="text-sm text-red-600">{error}</div>}
      {yaml && (
        <pre className="rounded bg-gray-900 text-gray-100 p-3 overflow-auto text-sm"><code>{yaml}</code></pre>
      )}
    </div>
  );
}
