import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export interface WorkerMeta {
  name: string;
  runtime: string; // e.g., python:3.12
  filePath: string;
  uploadedAt: string;
}

@Injectable()
export class MarketplaceService {
  private readonly logger = new Logger(MarketplaceService.name);
  private readonly storageDir = path.join(process.cwd(), 'uploads');
  private readonly registry: WorkerMeta[] = [];

  constructor() {
    if (!fs.existsSync(this.storageDir)) {
      fs.mkdirSync(this.storageDir, { recursive: true });
    }
  }

  list(): WorkerMeta[] {
    return this.registry.slice().reverse();
  }

  async saveAndRegister(tempPath: string, originalName: string, name: string, runtime: string): Promise<WorkerMeta> {
    const safeName = name?.replace(/[^a-zA-Z0-9-_]/g, '-') || path.parse(originalName).name;
    const dest = path.join(this.storageDir, `${Date.now()}-${originalName}`);

    await fs.promises.copyFile(tempPath, dest);

    const meta: WorkerMeta = {
      name: safeName,
      runtime: runtime || 'python:3.12',
      filePath: dest,
      uploadedAt: new Date().toISOString(),
    };
    this.registry.push(meta);
    this.logger.log(`Registered worker ${safeName} at ${dest}`);
    return meta;
  }
}

// TEST: Service registers uploaded workers and lists them
