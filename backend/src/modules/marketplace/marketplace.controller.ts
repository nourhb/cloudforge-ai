import { Controller, Get, Post, UploadedFile, UseInterceptors, Body, BadRequestException, UseGuards } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import * as path from 'path';
import * as fs from 'fs';
import type { Request } from 'express';
import { MarketplaceService } from './marketplace.service';
import { JwtAuthGuard } from '../../security/jwt.guard';

@Controller('/api/marketplace')
export class MarketplaceController {
  constructor(private readonly svc: MarketplaceService) {}

  @Get('list')
  list() {
    return { ok: true, items: this.svc.list() };
  }

  @Post('upload')
  @UseGuards(JwtAuthGuard)
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: (_req, _file, cb) => {
          const dest = path.join(process.cwd(), 'tmp-uploads');
          if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
          cb(null, dest);
        },
        filename: (_req: Request, file: Express.Multer.File, cb: (error: any, filename: string) => void) => {
          const unique = `${Date.now()}-${file.originalname}`;
          cb(null, unique);
        },
      }),
      limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
    })
  )
  async upload(@UploadedFile() file: Express.Multer.File, @Body() body: any) {
    if (!file) throw new BadRequestException('file is required');
    const name = body?.name || '';
    const runtime = body?.runtime || 'python:3.12';
    const meta = await this.svc.saveAndRegister(file.path, file.originalname, name, runtime);
    return { ok: true, item: meta };
  }
}

// TEST: Supertest can upload and then list returns item
