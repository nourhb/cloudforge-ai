import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
import { Request } from 'express';
import { JwtServiceLite } from './jwt.service';

@Injectable()
export class JwtAuthGuard implements CanActivate {
  constructor(private readonly jwt: JwtServiceLite) {}

  canActivate(context: ExecutionContext): boolean {
    const req = context.switchToHttp().getRequest<Request>();
    const auth = req.header('authorization') || req.header('Authorization');
    if (!auth || !auth.toLowerCase().startsWith('bearer ')) {
      throw new UnauthorizedException('missing bearer token');
    }
    const token = auth.slice(7).trim();
    const res = this.jwt.verify(token);
    if (!res.ok) {
      throw new UnauthorizedException(res.error || 'invalid token');
    }
    (req as any).user = res.payload;
    return true;
  }
}
