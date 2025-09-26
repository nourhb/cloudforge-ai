import { Module } from '@nestjs/common';
import { HealthController } from './health/health.controller';
import { MigrationController } from './modules/migration/migration.controller';
import { MigrationService } from './modules/migration/migration.service';

@Module({
  imports: [],
  controllers: [HealthController, MigrationController],
  providers: [MigrationService],
})
export class AppModule {}

// TEST: Module compiles
