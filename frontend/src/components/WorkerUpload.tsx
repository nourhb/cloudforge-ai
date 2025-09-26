"use client";

import * as React from 'react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

export function WorkerUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [runtime, setRuntime] = useState('python:3.12');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setMessage('Please choose a file');
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('name', name);
      form.append('runtime', runtime);
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'}/api/marketplace/upload`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.errors?.join(', ') || 'Upload failed');
      setMessage(`Uploaded ${data?.item?.name || name}`);
    } catch (err: any) {
      setMessage(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <div>
        <label className="block text-sm font-medium">Name</label>
        <input className="mt-1 w-full rounded border px-3 py-2" value={name} onChange={e => setName(e.target.value)} placeholder="echo-api" />
      </div>
      <div>
        <label className="block text-sm font-medium">Runtime</label>
        <select className="mt-1 w-full rounded border px-3 py-2" value={runtime} onChange={e => setRuntime(e.target.value)}>
          <option value="python:3.12">python:3.12</option>
          <option value="node:20-alpine">node:20-alpine</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium">File</label>
        <input className="mt-1 w-full" type="file" onChange={e => setFile(e.target.files?.[0] || null)} />
      </div>
      <div className="flex items-center gap-2">
        <Button type="submit" disabled={loading}>{loading ? 'Uploading...' : 'Upload Worker'}</Button>
        {message && <span className="text-sm text-gray-600">{message}</span>}
      </div>
    </form>
  );
}

// TEST: Upload form posts to backend and displays result
