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
  private refreshSecret = process.env.JWT_REFRESH_SECRET || 'refresh-secret-change-me';

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

  /**
   * Generate access token (short-lived)
   */
  generateToken(payload: Record<string, any>): string {
    return this.sign(payload, 15 * 60); // 15 minutes
  }

  /**
   * Generate refresh token (long-lived)
   */
  generateRefreshToken(payload: Record<string, any>): string {
    const header = { alg: this.algorithm, typ: 'JWT' };
    const now = Math.floor(Date.now() / 1000);
    const fullPayload = { 
      iat: now, 
      exp: now + (30 * 24 * 60 * 60), // 30 days
      type: 'refresh',
      ...payload 
    };

    const headerB64 = base64url(JSON.stringify(header));
    const payloadB64 = base64url(JSON.stringify(fullPayload));
    const signingInput = `${headerB64}.${payloadB64}`;
    const signature = crypto
      .createHmac('sha256', this.refreshSecret)
      .update(signingInput)
      .digest('base64')
      .replace(/=/g, '')
      .replace(/\+/g, '-')
      .replace(/\//g, '_');
    return `${signingInput}.${signature}`;
  }

  /**
   * Verify refresh token
   */
  verifyRefreshToken(token: string): any {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) throw new Error('Invalid token format');
      
      const [headerB64, payloadB64, signature] = parts;
      const signingInput = `${headerB64}.${payloadB64}`;
      const expected = crypto
        .createHmac('sha256', this.refreshSecret)
        .update(signingInput)
        .digest('base64')
        .replace(/=/g, '')
        .replace(/\+/g, '-')
        .replace(/\//g, '_');
      
      if (expected !== signature) throw new Error('Invalid signature');
      
      const payloadJson = Buffer.from(payloadB64.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
      const payload = JSON.parse(payloadJson);
      const now = Math.floor(Date.now() / 1000);
      
      if (payload.exp && now > payload.exp) throw new Error('Token expired');
      if (payload.type !== 'refresh') throw new Error('Invalid token type');
      
      return payload;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Extract payload without verification (for debugging)
   */
  decode(token: string): any {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      
      const payloadB64 = parts[1];
      const payloadJson = Buffer.from(payloadB64.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
      return JSON.parse(payloadJson);
    } catch {
      return null;
    }
  }
}
