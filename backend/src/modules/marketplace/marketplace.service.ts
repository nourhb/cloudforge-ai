import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const execAsync = promisify(exec);

export interface WorkerMeta {
  id: string;
  name: string;
  runtime: string; // e.g., python:3.12
  filePath: string;
  uploadedAt: string;
  version: string;
  description?: string;
  author?: string;
  tags: string[];
  status: 'active' | 'inactive' | 'deprecated';
  downloadCount: number;
  rating: number;
  reviews: WorkerReview[];
  dependencies: string[];
  permissions: string[];
  size: number;
  checksum: string;
  category: WorkerCategory;
  lastUpdated: string;
  documentation?: string;
  examples?: WorkerExample[];
}

export interface WorkerReview {
  id: string;
  userId: string;
  username: string;
  rating: number;
  comment: string;
  timestamp: string;
  helpful: number;
}

export interface WorkerExample {
  title: string;
  description: string;
  code: string;
  expectedOutput?: string;
}

export enum WorkerCategory {
  DATA_PROCESSING = 'data-processing',
  ML_AI = 'ml-ai',
  AUTOMATION = 'automation',
  MONITORING = 'monitoring',
  SECURITY = 'security',
  NETWORKING = 'networking',
  UTILITIES = 'utilities',
  INTEGRATION = 'integration'
}

export interface WorkerSearchParams {
  query?: string;
  category?: WorkerCategory;
  runtime?: string;
  tags?: string[];
  author?: string;
  minRating?: number;
  sortBy?: 'name' | 'rating' | 'downloads' | 'updated';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export interface WorkerExecutionRequest {
  workerId: string;
  inputs: Record<string, any>;
  timeout?: number;
  environment?: Record<string, string>;
}

export interface WorkerExecutionResult {
  success: boolean;
  output: any;
  logs: string[];
  errors: string[];
  duration: number;
  timestamp: string;
  resources: {
    memoryUsed: number;
    cpuTime: number;
  };
}

export interface WorkerStats {
  totalWorkers: number;
  activeWorkers: number;
  categoryCounts: Record<string, number>;
  popularTags: Array<{ tag: string; count: number }>;
  topRated: WorkerMeta[];
  mostDownloaded: WorkerMeta[];
  recentlyAdded: WorkerMeta[];
}

@Injectable()
export class MarketplaceService {
  private readonly logger = new Logger(MarketplaceService.name);
  private readonly storageDir = path.join(process.cwd(), 'uploads');
  private readonly registry: WorkerMeta[] = [];
  private readonly executionHistory = new Map<string, WorkerExecutionResult[]>();

  constructor() {
    if (!fs.existsSync(this.storageDir)) {
      fs.mkdirSync(this.storageDir, { recursive: true });
    }
    this.initializeDefaultWorkers();
  }

  /**
   * Initialize marketplace with default AI workers
   */
  private initializeDefaultWorkers(): void {
    const defaultWorkers: Partial<WorkerMeta>[] = [
      {
        name: 'ai-migration-analyzer',
        runtime: 'python:3.12',
        description: 'AI-powered database migration analysis and planning',
        category: WorkerCategory.ML_AI,
        tags: ['migration', 'database', 'ai', 'analysis'],
        author: 'CloudForge AI',
        rating: 4.8,
        downloadCount: 1250,
        status: 'active'
      },
      {
        name: 'iac-generator',
        runtime: 'python:3.12',
        description: 'Generate Infrastructure as Code from natural language',
        category: WorkerCategory.AUTOMATION,
        tags: ['iac', 'kubernetes', 'terraform', 'automation'],
        author: 'CloudForge AI',
        rating: 4.6,
        downloadCount: 890,
        status: 'active'
      },
      {
        name: 'anomaly-detector',
        runtime: 'python:3.12',
        description: 'Real-time anomaly detection for system metrics',
        category: WorkerCategory.MONITORING,
        tags: ['anomaly', 'monitoring', 'ml', 'alerting'],
        author: 'CloudForge AI',
        rating: 4.7,
        downloadCount: 1100,
        status: 'active'
      },
      {
        name: 'forecasting-engine',
        runtime: 'python:3.12',
        description: 'Advanced time series forecasting for capacity planning',
        category: WorkerCategory.ML_AI,
        tags: ['forecasting', 'timeseries', 'prediction', 'planning'],
        author: 'CloudForge AI',
        rating: 4.5,
        downloadCount: 750,
        status: 'active'
      }
    ];

    defaultWorkers.forEach(worker => {
      const fullWorker: WorkerMeta = {
        id: crypto.randomUUID(),
        name: worker.name!,
        runtime: worker.runtime!,
        filePath: path.join(__dirname, '..', '..', '..', 'ai-scripts', `${worker.name}.py`),
        uploadedAt: new Date().toISOString(),
        version: '1.0.0',
        description: worker.description,
        author: worker.author,
        tags: worker.tags || [],
        status: worker.status as any || 'active',
        downloadCount: worker.downloadCount || 0,
        rating: worker.rating || 0,
        reviews: [],
        dependencies: ['numpy', 'pandas', 'scikit-learn'],
        permissions: ['read', 'compute'],
        size: 0,
        checksum: '',
        category: worker.category!,
        lastUpdated: new Date().toISOString(),
        examples: []
      };
      this.registry.push(fullWorker);
    });

    this.logger.log(`Initialized marketplace with ${defaultWorkers.length} default workers`);
  }

  /**
   * List workers with optional filtering and pagination
   */
  list(params: WorkerSearchParams = {}): { workers: WorkerMeta[]; total: number; offset: number; limit: number } {
    let filtered = [...this.registry];

    // Apply filters
    if (params.query) {
      const query = params.query.toLowerCase();
      filtered = filtered.filter(w => 
        w.name.toLowerCase().includes(query) ||
        w.description?.toLowerCase().includes(query) ||
        w.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    if (params.category) {
      filtered = filtered.filter(w => w.category === params.category);
    }

    if (params.runtime) {
      filtered = filtered.filter(w => w.runtime === params.runtime);
    }

    if (params.tags && params.tags.length > 0) {
      filtered = filtered.filter(w => 
        params.tags!.some(tag => w.tags.includes(tag))
      );
    }

    if (params.author) {
      filtered = filtered.filter(w => w.author === params.author);
    }

    if (params.minRating) {
      filtered = filtered.filter(w => w.rating >= params.minRating!);
    }

    // Apply sorting
    const sortBy = params.sortBy || 'rating';
    const sortOrder = params.sortOrder || 'desc';
    
    filtered.sort((a, b) => {
      let aVal: any, bVal: any;
      
      switch (sortBy) {
        case 'name':
          aVal = a.name.toLowerCase();
          bVal = b.name.toLowerCase();
          break;
        case 'rating':
          aVal = a.rating;
          bVal = b.rating;
          break;
        case 'downloads':
          aVal = a.downloadCount;
          bVal = b.downloadCount;
          break;
        case 'updated':
          aVal = new Date(a.lastUpdated).getTime();
          bVal = new Date(b.lastUpdated).getTime();
          break;
        default:
          aVal = a.rating;
          bVal = b.rating;
      }
      
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    // Apply pagination
    const offset = params.offset || 0;
    const limit = params.limit || 20;
    const paginated = filtered.slice(offset, offset + limit);

    return {
      workers: paginated,
      total: filtered.length,
      offset,
      limit
    };
  }

  /**
   * Get worker by ID
   */
  getWorker(id: string): WorkerMeta {
    const worker = this.registry.find(w => w.id === id);
    if (!worker) {
      throw new HttpException('Worker not found', HttpStatus.NOT_FOUND);
    }
    return worker;
  }

  /**
   * Save and register a new worker
   */
  async saveAndRegister(
    tempPath: string, 
    originalName: string, 
    metadata: Partial<WorkerMeta>
  ): Promise<WorkerMeta> {
    try {
      const safeName = metadata.name?.replace(/[^a-zA-Z0-9-_]/g, '-') || path.parse(originalName).name;
      const dest = path.join(this.storageDir, `${Date.now()}-${originalName}`);

      // Copy file and calculate checksum
      await fs.promises.copyFile(tempPath, dest);
      const fileStats = await fs.promises.stat(dest);
      const fileBuffer = await fs.promises.readFile(dest);
      const checksum = crypto.createHash('sha256').update(fileBuffer).digest('hex');

      const worker: WorkerMeta = {
        id: crypto.randomUUID(),
        name: safeName,
        runtime: metadata.runtime || 'python:3.12',
        filePath: dest,
        uploadedAt: new Date().toISOString(),
        version: metadata.version || '1.0.0',
        description: metadata.description || '',
        author: metadata.author || 'Anonymous',
        tags: metadata.tags || [],
        status: 'active',
        downloadCount: 0,
        rating: 0,
        reviews: [],
        dependencies: metadata.dependencies || [],
        permissions: metadata.permissions || ['read'],
        size: fileStats.size,
        checksum,
        category: metadata.category || WorkerCategory.UTILITIES,
        lastUpdated: new Date().toISOString(),
        documentation: metadata.documentation,
        examples: metadata.examples || []
      };

      this.registry.push(worker);
      this.logger.log(`Registered worker ${safeName} (${worker.id}) at ${dest}`);
      
      return worker;
    } catch (error) {
      this.logger.error('Failed to register worker', error);
      throw new HttpException('Failed to register worker', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Execute a worker with given inputs
   */
  async executeWorker(request: WorkerExecutionRequest): Promise<WorkerExecutionResult> {
    const startTime = Date.now();
    const worker = this.getWorker(request.workerId);
    
    try {
      this.logger.log(`Executing worker ${worker.name} (${worker.id})`);
      
      // Prepare execution environment
      const tempDir = path.join(this.storageDir, 'temp', crypto.randomUUID());
      await fs.promises.mkdir(tempDir, { recursive: true });
      
      // Write inputs to temporary file
      const inputFile = path.join(tempDir, 'input.json');
      await fs.promises.writeFile(inputFile, JSON.stringify(request.inputs, null, 2));
      
      // Execute the worker
      const timeout = request.timeout || 30000; // 30 seconds default
      const env = { ...process.env, ...request.environment };
      
      const { stdout, stderr } = await execAsync(
        `python "${worker.filePath}" "${inputFile}"`,
        { 
          timeout,
          env,
          cwd: tempDir
        }
      );
      
      // Parse output
      let output: any;
      try {
        output = JSON.parse(stdout);
      } catch {
        output = stdout;
      }
      
      const duration = Date.now() - startTime;
      const result: WorkerExecutionResult = {
        success: true,
        output,
        logs: stdout.split('\n').filter(line => line.trim()),
        errors: stderr ? stderr.split('\n').filter(line => line.trim()) : [],
        duration,
        timestamp: new Date().toISOString(),
        resources: {
          memoryUsed: 0, // Would need process monitoring
          cpuTime: duration
        }
      };
      
      // Store execution history
      if (!this.executionHistory.has(worker.id)) {
        this.executionHistory.set(worker.id, []);
      }
      this.executionHistory.get(worker.id)!.push(result);
      
      // Cleanup
      await fs.promises.rm(tempDir, { recursive: true, force: true });
      
      this.logger.log(`Worker ${worker.name} executed successfully in ${duration}ms`);
      return result;
      
    } catch (error) {
      const duration = Date.now() - startTime;
      const result: WorkerExecutionResult = {
        success: false,
        output: null,
        logs: [],
        errors: [error instanceof Error ? error.message : String(error)],
        duration,
        timestamp: new Date().toISOString(),
        resources: {
          memoryUsed: 0,
          cpuTime: duration
        }
      };
      
      this.logger.error(`Worker ${worker.name} execution failed`, error);
      return result;
    }
  }

  /**
   * Add a review for a worker
   */
  async addReview(workerId: string, review: Omit<WorkerReview, 'id' | 'timestamp' | 'helpful'>): Promise<WorkerReview> {
    const worker = this.getWorker(workerId);
    
    const newReview: WorkerReview = {
      id: crypto.randomUUID(),
      ...review,
      timestamp: new Date().toISOString(),
      helpful: 0
    };
    
    worker.reviews.push(newReview);
    
    // Recalculate rating
    const totalRating = worker.reviews.reduce((sum, r) => sum + r.rating, 0);
    worker.rating = totalRating / worker.reviews.length;
    
    this.logger.log(`Added review for worker ${worker.name} by ${review.username}`);
    return newReview;
  }

  /**
   * Update worker metadata
   */
  async updateWorker(id: string, updates: Partial<WorkerMeta>): Promise<WorkerMeta> {
    const worker = this.getWorker(id);
    
    // Apply allowed updates
    const allowedFields = ['description', 'tags', 'status', 'documentation', 'examples'];
    allowedFields.forEach(field => {
      if (updates[field as keyof WorkerMeta] !== undefined) {
        (worker as any)[field] = updates[field as keyof WorkerMeta];
      }
    });
    
    worker.lastUpdated = new Date().toISOString();
    
    this.logger.log(`Updated worker ${worker.name} (${id})`);
    return worker;
  }

  /**
   * Delete a worker
   */
  async deleteWorker(id: string): Promise<{ success: boolean; message: string }> {
    const index = this.registry.findIndex(w => w.id === id);
    if (index === -1) {
      throw new HttpException('Worker not found', HttpStatus.NOT_FOUND);
    }
    
    const worker = this.registry[index];
    
    try {
      // Remove file if it exists
      if (fs.existsSync(worker.filePath)) {
        await fs.promises.unlink(worker.filePath);
      }
      
      // Remove from registry
      this.registry.splice(index, 1);
      
      // Clean up execution history
      this.executionHistory.delete(id);
      
      this.logger.log(`Deleted worker ${worker.name} (${id})`);
      return { success: true, message: 'Worker deleted successfully' };
      
    } catch (error) {
      this.logger.error(`Failed to delete worker ${id}`, error);
      throw new HttpException('Failed to delete worker', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Get marketplace statistics
   */
  getStats(): WorkerStats {
    const workers = this.registry;
    
    // Category counts
    const categoryCounts: Record<string, number> = {};
    Object.values(WorkerCategory).forEach(cat => {
      categoryCounts[cat] = workers.filter(w => w.category === cat).length;
    });
    
    // Popular tags
    const tagCounts: Record<string, number> = {};
    workers.forEach(w => {
      w.tags.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });
    const popularTags = Object.entries(tagCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([tag, count]) => ({ tag, count }));
    
    // Top rated (min 3 reviews)
    const topRated = workers
      .filter(w => w.reviews.length >= 3)
      .sort((a, b) => b.rating - a.rating)
      .slice(0, 5);
    
    // Most downloaded
    const mostDownloaded = workers
      .sort((a, b) => b.downloadCount - a.downloadCount)
      .slice(0, 5);
    
    // Recently added
    const recentlyAdded = workers
      .sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime())
      .slice(0, 5);
    
    return {
      totalWorkers: workers.length,
      activeWorkers: workers.filter(w => w.status === 'active').length,
      categoryCounts,
      popularTags,
      topRated,
      mostDownloaded,
      recentlyAdded
    };
  }

  /**
   * Get worker execution history
   */
  getExecutionHistory(workerId: string): WorkerExecutionResult[] {
    return this.executionHistory.get(workerId) || [];
  }

  /**
   * Download worker (increment download count)
   */
  async downloadWorker(id: string): Promise<{ filePath: string; worker: WorkerMeta }> {
    const worker = this.getWorker(id);
    
    if (!fs.existsSync(worker.filePath)) {
      throw new HttpException('Worker file not found', HttpStatus.NOT_FOUND);
    }
    
    // Increment download count
    worker.downloadCount++;
    
    this.logger.log(`Worker ${worker.name} downloaded. Total downloads: ${worker.downloadCount}`);
    
    return {
      filePath: worker.filePath,
      worker
    };
  }
}

// TEST: Service registers uploaded workers and lists them
