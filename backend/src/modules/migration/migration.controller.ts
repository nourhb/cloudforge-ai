import { Body, Controller, HttpException, HttpStatus, Post } from '@nestjs/common';
import { MigrationRequest, MigrationService } from './migration.service';

@Controller('/api/migration')
export class MigrationController {
  constructor(private readonly migration: MigrationService) {}

  @Post('migrate')
  async migrate(@Body() body: MigrationRequest) {
    const res = await this.migration.migrate(body);
    if (!res.ok) {
      throw new HttpException({ ok: false, errors: res.errors }, HttpStatus.BAD_REQUEST);
    }
    return res;
  }
}

// TEST: Supertest e2e will validate /api/migration/migrate
