import { 
  Body, 
  Controller, 
  Get, 
  Post, 
  Delete,
  Param,
  Query,
  HttpException, 
  HttpStatus,
  Logger,
  UseGuards,
  UsePipes,
  ValidationPipe
} from '@nestjs/common';
import { 
  MigrationRequest, 
  MigrationService, 
  MigrationResult,
  AIMigrationAnalysis 
} from './migration.service';

export interface MigrationAnalysisDto {
  source: string;
  target: string;
  tables?: string[];
  options?: any;
}

export interface MigrationStatusQuery {
  migrationId?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

@Controller('/api/migration')
@UsePipes(new ValidationPipe({ transform: true }))
export class MigrationController {
  private readonly logger = new Logger(MigrationController.name);

  constructor(private readonly migration: MigrationService) {}

  /**
   * Analyze migration requirements using AI
   */
  @Post('analyze')
  async analyzeMigration(@Body() body: MigrationAnalysisDto): Promise<AIMigrationAnalysis> {
    try {
      this.logger.log(`Migration analysis requested: ${body.source} -> ${body.target}`);
      return await this.migration.analyzeMigration(body as MigrationRequest);
    } catch (error) {
      this.logger.error('Migration analysis failed', error);
      throw new HttpException(
        { message: 'Migration analysis failed', error: error instanceof Error ? error.message : String(error) },
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  /**
   * Execute migration
   */
  @Post('migrate')
  async migrate(@Body() body: MigrationRequest): Promise<MigrationResult> {
    try {
      this.logger.log(`Migration requested: ${body.source} -> ${body.target}`);
      const result = await this.migration.migrate(body);
      
      if (!result.ok) {
        throw new HttpException(
          { 
            message: 'Migration failed', 
            errors: result.errors,
            migrationId: result.migrationId,
            summary: result.summary
          }, 
          HttpStatus.BAD_REQUEST
        );
      }
      
      return result;
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      
      this.logger.error('Migration execution failed', error);
      throw new HttpException(
        { message: 'Migration execution failed', error: error instanceof Error ? error.message : String(error) },
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  /**
   * Get migration status by ID
   */
  @Get('status/:migrationId')
  async getMigrationStatus(@Param('migrationId') migrationId: string) {
    try {
      return await this.migration.getMigrationStatus(migrationId);
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Failed to get migration status', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * List all migrations with optional filtering
   */
  @Get('list')
  async listMigrations(@Query() query: MigrationStatusQuery) {
    try {
      const migrations = await this.migration.listMigrations();
      
      // Apply filters
      let filtered = migrations;
      
      if (query.status) {
        filtered = filtered.filter(m => m.status === query.status);
      }
      
      if (query.migrationId) {
        filtered = filtered.filter(m => m.migrationId.includes(query.migrationId!));
      }
      
      // Apply pagination
      const offset = query.offset || 0;
      const limit = query.limit || 50;
      const paginated = filtered.slice(offset, offset + limit);
      
      return {
        migrations: paginated,
        total: filtered.length,
        offset,
        limit
      };
    } catch (error) {
      this.logger.error('Failed to list migrations', error);
      throw new HttpException('Failed to list migrations', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Cancel an active migration
   */
  @Delete('cancel/:migrationId')
  async cancelMigration(@Param('migrationId') migrationId: string) {
    try {
      return await this.migration.cancelMigration(migrationId);
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Failed to cancel migration', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Validate migration configuration without executing
   */
  @Post('validate')
  async validateMigration(@Body() body: MigrationRequest) {
    try {
      const validationRequest = {
        ...body,
        options: {
          ...body.options,
          validateOnly: true
        }
      };
      
      return await this.migration.migrate(validationRequest);
    } catch (error) {
      this.logger.error('Migration validation failed', error);
      throw new HttpException(
        { message: 'Migration validation failed', error: error instanceof Error ? error.message : String(error) },
        HttpStatus.BAD_REQUEST
      );
    }
  }

  /**
   * Get migration health and service status
   */
  @Get('health')
  async getHealth() {
    try {
      const activeMigrations = await this.migration.listMigrations();
      const runningCount = activeMigrations.filter(m => m.status === 'running').length;
      
      return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        activeMigrations: runningCount,
        totalMigrations: activeMigrations.length,
        service: 'migration-service',
        version: '1.0.0'
      };
    } catch (error) {
      throw new HttpException(
        { status: 'unhealthy', error: error instanceof Error ? error.message : String(error) },
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }
}

// TEST: Supertest e2e will validate /api/migration/migrate
