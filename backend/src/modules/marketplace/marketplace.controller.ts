import { 
  Controller, 
  Get, 
  Post, 
  Put,
  Delete,
  Param,
  Query,
  UploadedFile, 
  UseInterceptors, 
  Body, 
  BadRequestException, 
  UseGuards,
  Logger,
  HttpException,
  HttpStatus,
  Res,
  UsePipes,
  ValidationPipe
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { Response } from 'express';
import * as path from 'path';
import * as fs from 'fs';
import type { Request } from 'express';
import { 
  MarketplaceService, 
  WorkerMeta, 
  WorkerSearchParams,
  WorkerExecutionRequest,
  WorkerCategory
} from './marketplace.service';
import { JwtAuthGuard } from '../../security/jwt.guard';

export interface UploadWorkerDto {
  name: string;
  description?: string;
  author?: string;
  tags?: string;
  category?: WorkerCategory;
  version?: string;
  runtime?: string;
  dependencies?: string;
  permissions?: string;
  documentation?: string;
}

export interface WorkerReviewDto {
  userId: string;
  username: string;
  rating: number;
  comment: string;
}

@Controller('/api/marketplace')
@UsePipes(new ValidationPipe({ transform: true }))
export class MarketplaceController {
  private readonly logger = new Logger(MarketplaceController.name);

  constructor(private readonly svc: MarketplaceService) {}

  /**
   * List workers with filtering and pagination
   */
  @Get('list')
  list(@Query() query: WorkerSearchParams) {
    try {
      const result = this.svc.list(query);
      return { ok: true, ...result };
    } catch (error) {
      this.logger.error('Failed to list workers', error);
      throw new HttpException('Failed to list workers', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Get worker by ID
   */
  @Get('worker/:id')
  getWorker(@Param('id') id: string) {
    try {
      const worker = this.svc.getWorker(id);
      return { ok: true, worker };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Failed to get worker', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Get marketplace statistics
   */
  @Get('stats')
  getStats() {
    try {
      const stats = this.svc.getStats();
      return { ok: true, stats };
    } catch (error) {
      this.logger.error('Failed to get marketplace stats', error);
      throw new HttpException('Failed to get stats', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Search workers
   */
  @Get('search')
  search(@Query() query: WorkerSearchParams) {
    try {
      const result = this.svc.list(query);
      return { 
        ok: true, 
        results: result.workers,
        pagination: {
          total: result.total,
          offset: result.offset,
          limit: result.limit,
          hasMore: result.offset + result.limit < result.total
        }
      };
    } catch (error) {
      this.logger.error('Worker search failed', error);
      throw new HttpException('Search failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Upload and register a new worker
   */
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
      limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
      fileFilter: (_req, file, cb) => {
        // Allow Python files and archives
        const allowedTypes = ['.py', '.zip', '.tar.gz', '.tar'];
        const ext = path.extname(file.originalname).toLowerCase();
        if (allowedTypes.includes(ext)) {
          cb(null, true);
        } else {
          cb(new BadRequestException('Only Python files and archives are allowed'), false);
        }
      }
    })
  )
  async upload(@UploadedFile() file: Express.Multer.File, @Body() body: UploadWorkerDto) {
    if (!file) {
      throw new BadRequestException('File is required');
    }

    try {
      // Parse complex fields
      const metadata: Partial<WorkerMeta> = {
        name: body.name,
        description: body.description,
        author: body.author,
        tags: body.tags ? body.tags.split(',').map(tag => tag.trim()) : [],
        category: body.category || WorkerCategory.UTILITIES,
        version: body.version || '1.0.0',
        runtime: body.runtime || 'python:3.12',
        dependencies: body.dependencies ? body.dependencies.split(',').map(dep => dep.trim()) : [],
        permissions: body.permissions ? body.permissions.split(',').map(perm => perm.trim()) : ['read'],
        documentation: body.documentation
      };

      const worker = await this.svc.saveAndRegister(file.path, file.originalname, metadata);
      
      this.logger.log(`Worker uploaded successfully: ${worker.name} (${worker.id})`);
      return { ok: true, worker };
      
    } catch (error) {
      this.logger.error('Worker upload failed', error);
      throw new HttpException('Upload failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Execute a worker
   */
  @Post('execute')
  @UseGuards(JwtAuthGuard)
  async executeWorker(@Body() request: WorkerExecutionRequest) {
    try {
      if (!request.workerId) {
        throw new BadRequestException('Worker ID is required');
      }

      const result = await this.svc.executeWorker(request);
      return { ok: true, result };
      
    } catch (error) {
      this.logger.error('Worker execution failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Execution failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Download a worker file
   */
  @Get('download/:id')
  @UseGuards(JwtAuthGuard)
  async downloadWorker(@Param('id') id: string, @Res() res: Response) {
    try {
      const { filePath, worker } = await this.svc.downloadWorker(id);
      
      res.setHeader('Content-Disposition', `attachment; filename="${worker.name}.py"`);
      res.setHeader('Content-Type', 'application/octet-stream');
      
      return res.sendFile(filePath);
      
    } catch (error) {
      this.logger.error('Worker download failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Download failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Add a review for a worker
   */
  @Post('review/:id')
  @UseGuards(JwtAuthGuard)
  async addReview(@Param('id') id: string, @Body() reviewDto: WorkerReviewDto) {
    try {
      if (!reviewDto.userId || !reviewDto.username || !reviewDto.rating) {
        throw new BadRequestException('User ID, username, and rating are required');
      }

      if (reviewDto.rating < 1 || reviewDto.rating > 5) {
        throw new BadRequestException('Rating must be between 1 and 5');
      }

      const review = await this.svc.addReview(id, reviewDto);
      return { ok: true, review };
      
    } catch (error) {
      this.logger.error('Add review failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Failed to add review', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Update worker metadata
   */
  @Put('worker/:id')
  @UseGuards(JwtAuthGuard)
  async updateWorker(@Param('id') id: string, @Body() updates: Partial<WorkerMeta>) {
    try {
      const worker = await this.svc.updateWorker(id, updates);
      return { ok: true, worker };
      
    } catch (error) {
      this.logger.error('Worker update failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Update failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Delete a worker
   */
  @Delete('worker/:id')
  @UseGuards(JwtAuthGuard)
  async deleteWorker(@Param('id') id: string) {
    try {
      const result = await this.svc.deleteWorker(id);
      return { ok: true, ...result };
      
    } catch (error) {
      this.logger.error('Worker deletion failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Deletion failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Get worker execution history
   */
  @Get('worker/:id/history')
  @UseGuards(JwtAuthGuard)
  getExecutionHistory(@Param('id') id: string) {
    try {
      const history = this.svc.getExecutionHistory(id);
      return { ok: true, history };
      
    } catch (error) {
      this.logger.error('Failed to get execution history', error);
      throw new HttpException('Failed to get history', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Get available worker categories
   */
  @Get('categories')
  getCategories() {
    return { 
      ok: true, 
      categories: Object.values(WorkerCategory).map(cat => ({
        value: cat,
        label: cat.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())
      }))
    };
  }

  /**
   * Get marketplace health status
   */
  @Get('health')
  getHealth() {
    try {
      const stats = this.svc.getStats();
      return {
        ok: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        workers: {
          total: stats.totalWorkers,
          active: stats.activeWorkers
        },
        service: 'marketplace-service',
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

// TEST: Supertest can upload and then list returns item
