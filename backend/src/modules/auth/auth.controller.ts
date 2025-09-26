import { Body, Controller, HttpException, HttpStatus, Post } from '@nestjs/common';
import { AuthService } from './auth.service';
import { JwtServiceLite } from '../../security/jwt.service';

@Controller('/api/auth')
export class AuthController {
  constructor(private readonly auth: AuthService, private readonly jwt: JwtServiceLite) {}

  @Post('request-otp')
  requestOtp(@Body() body: { identifier: string }) {
    const identifier = body?.identifier?.trim();
    if (!identifier) {
      throw new HttpException({ ok: false, error: 'identifier required' }, HttpStatus.BAD_REQUEST);
    }
    const code = this.auth.generateOtp(identifier);
    // In production, send via email/SMS. Here we return code for demo/testing only.
    return { ok: true, sent: true, code };
  }

  @Post('verify')
  verify(@Body() body: { identifier: string; code: string }) {
    const identifier = body?.identifier?.trim();
    const code = body?.code?.trim();
    if (!identifier || !code) {
      throw new HttpException({ ok: false, error: 'identifier and code required' }, HttpStatus.BAD_REQUEST);
    }
    const ok = this.auth.verifyOtp(identifier, code);
    if (!ok) {
      throw new HttpException({ ok: false, error: 'invalid or expired code' }, HttpStatus.UNAUTHORIZED);
    }
    const token = this.jwt.sign({ sub: identifier });
    return { ok: true, token };
  }
}
