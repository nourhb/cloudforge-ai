"use client";

import * as React from 'react';
import { useState } from 'react';

export default function LoginPage() {
  const [identifier, setIdentifier] = useState("");
  const [code, setCode] = useState("");
  const [phase, setPhase] = useState<"request" | "verify">("request");
  const [msg, setMsg] = useState<string | null>(null);
  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

  const requestOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    try {
      const res = await fetch(`${base}/api/auth/request-otp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier }),
      });
      const data = await res.json();
      if (!res.ok || !data?.ok) throw new Error(data?.error || "failed");
      setMsg(`OTP sent (demo code: ${data.code})`);
      setPhase("verify");
    } catch (e: any) {
      setMsg(e.message || "failed");
    }
  };

  const verifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    try {
      const res = await fetch(`${base}/api/auth/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, code }),
      });
      const data = await res.json();
      if (!res.ok || !data?.ok) throw new Error(data?.error || "failed");
      setMsg(`Logged in with token: ${data.token}`);
    } catch (e: any) {
      setMsg(e.message || "failed");
    }
  };

  return (
    <div className="mx-auto max-w-md p-6 space-y-4">
      <h1 className="text-3xl font-bold">Login (OTP)</h1>
      {phase === "request" && (
        <form onSubmit={requestOtp} className="space-y-3">
          <div>
            <label className="block text-sm font-medium">Identifier (email/phone)</label>
            <input className="mt-1 w-full rounded border px-3 py-2" value={identifier} onChange={(e) => setIdentifier(e.target.value)} placeholder="you@example.com" />
          </div>
          <button type="submit" className="rounded bg-blue-600 px-4 py-2 text-white">Request OTP</button>
        </form>
      )}
      {phase === "verify" && (
        <form onSubmit={verifyOtp} className="space-y-3">
          <div>
            <label className="block text-sm font-medium">Code</label>
            <input className="mt-1 w-full rounded border px-3 py-2" value={code} onChange={(e) => setCode(e.target.value)} placeholder="6-digit code" />
          </div>
          <button type="submit" className="rounded bg-green-600 px-4 py-2 text-white">Verify</button>
        </form>
      )}
      {msg && <div className="text-sm text-gray-700">{msg}</div>}
    </div>
  );
}
