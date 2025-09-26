import { Injectable, Logger } from '@nestjs/common';

interface OtpEntry {
  code: string;
  expiresAt: number;
}

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);
  private store = new Map<string, OtpEntry>();
  private ttlMs = 5 * 60 * 1000; // 5 minutes

  generateOtp(identifier: string): string {
    const code = String(Math.floor(100000 + Math.random() * 900000));
    const expiresAt = Date.now() + this.ttlMs;
    this.store.set(identifier, { code, expiresAt });
    this.logger.log(`OTP generated for ${identifier}`);
    return code;
  }

  verifyOtp(identifier: string, code: string): boolean {
    const entry = this.store.get(identifier);
    if (!entry) return false;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(identifier);
      return false;
    }
    const ok = entry.code === code;
    if (ok) this.store.delete(identifier);
    return ok;
  }
}
