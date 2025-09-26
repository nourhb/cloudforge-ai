import { Injectable } from '@nestjs/common';
import crypto from 'crypto';

// Lightweight JWT HS256 implementation to avoid adding runtime deps
// NOTE: For production, keep JWT_SECRET safe and rotate regularly.

function base64url(input: Buffer | string) {
  const buf = Buffer.isBuffer(input) ? input : Buffer.from(input);
  return buf.toString('base64').replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
}

@Injectable()
export class JwtServiceLite {
  private algorithm = 'HS256';
  private secret = process.env.JWT_SECRET || 'dev-secret-change-me';

  sign(payload: Record<string, any>, expiresInSeconds = 3600): string {
    const header = { alg: this.algorithm, typ: 'JWT' };
    const now = Math.floor(Date.now() / 1000);
    const fullPayload = { iat: now, exp: now + expiresInSeconds, ...payload };

    const headerB64 = base64url(JSON.stringify(header));
    const payloadB64 = base64url(JSON.stringify(fullPayload));
    const signingInput = `${headerB64}.${payloadB64}`;
    const signature = crypto
      .createHmac('sha256', this.secret)
      .update(signingInput)
      .digest('base64')
      .replace(/=/g, '')
      .replace(/\+/g, '-')
      .replace(/\//g, '_');
    return `${signingInput}.${signature}`;
  }

  verify(token: string): { ok: boolean; payload?: any; error?: string } {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return { ok: false, error: 'invalid token' };
      const [headerB64, payloadB64, signature] = parts;
      const signingInput = `${headerB64}.${payloadB64}`;
      const expected = crypto
        .createHmac('sha256', this.secret)
        .update(signingInput)
        .digest('base64')
        .replace(/=/g, '')
        .replace(/\+/g, '-')
        .replace(/\//g, '_');
      if (expected !== signature) return { ok: false, error: 'signature mismatch' };
      const payloadJson = Buffer.from(payloadB64.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
      const payload = JSON.parse(payloadJson);
      const now = Math.floor(Date.now() / 1000);
      if (payload.exp && now > payload.exp) return { ok: false, error: 'expired' };
      return { ok: true, payload };
    } catch (e: any) {
      return { ok: false, error: e?.message || 'verify failed' };
    }
  }
}
