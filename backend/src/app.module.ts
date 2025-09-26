import { Module } from '@nestjs/common';
import { HealthController } from './health/health.controller';
import { MigrationController } from './modules/migration/migration.controller';
import { MigrationService } from './modules/migration/migration.service';
import { MarketplaceController } from './modules/marketplace/marketplace.controller';
import { MarketplaceService } from './modules/marketplace/marketplace.service';
import { IacController } from './modules/iac/iac.controller';
import { AuthController } from './modules/auth/auth.controller';
import { AuthService } from './modules/auth/auth.service';
import { MetricsController } from './metrics/metrics.controller';
import { JwtServiceLite } from './security/jwt.service';

@Module({
  imports: [],
  controllers: [HealthController, MigrationController, MarketplaceController, IacController, AuthController, MetricsController],
  providers: [MigrationService, MarketplaceService, AuthService, JwtServiceLite],
})
export class AppModule {}

// TEST: Module compiles
