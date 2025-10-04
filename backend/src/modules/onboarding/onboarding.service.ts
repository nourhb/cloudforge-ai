import { Injectable, NotFoundException, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export interface TrainingProgress {
  id?: number;
  userId: string;
  moduleId: string;
  status: 'not_started' | 'in_progress' | 'completed';
  progress: number; // 0-100
  startedAt?: Date;
  completedAt?: Date;
  timeSpent: number; // minutes
  lastAccessed: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface UserStats {
  userId: string;
  totalModules: number;
  completedModules: number;
  inProgressModules: number;
  totalTimeSpent: number;
  averageScore: number;
  currentStreak: number;
  lastLoginDate: Date;
  joinedDate: Date;
}

export interface TrainingModule {
  id: string;
  title: string;
  description: string;
  content: string;
  duration: number; // estimated minutes
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  type: 'video' | 'interactive' | 'reading' | 'hands-on';
  prerequisites: string[];
  tags: string[];
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

// Simple JSON-based storage for development (replace with real database in production)
interface StorageData {
  modules: TrainingModule[];
  progress: TrainingProgress[];
  stats: UserStats[];
}

@Injectable()
export class OnboardingService implements OnModuleInit {
  private dataPath: string;
  private storage: StorageData = {
    modules: [],
    progress: [],
    stats: []
  };

  constructor() {
    this.dataPath = path.join(process.cwd(), 'data', 'onboarding-data.json');
  }

  async onModuleInit() {
    await this.initializeDatabase();
  }

  private async initializeDatabase() {
    // Ensure data directory exists
    const dataDir = path.dirname(this.dataPath);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Load existing data or create new
    if (fs.existsSync(this.dataPath)) {
      try {
        const rawData = fs.readFileSync(this.dataPath, 'utf8');
        this.storage = JSON.parse(rawData);
      } catch (error) {
        console.error('Error loading onboarding data:', error);
        await this.seedInitialData();
      }
    } else {
      await this.seedInitialData();
    }
  }

  private async saveData() {
    try {
      fs.writeFileSync(this.dataPath, JSON.stringify(this.storage, null, 2));
    } catch (error) {
      console.error('Error saving onboarding data:', error);
    }
  }

  private async seedInitialData() {
    // Initialize with sample training modules
    const now = new Date();
    this.storage.modules = [
      {
        id: 'intro-cloudforge',
        title: 'Introduction to CloudForge AI',
        description: 'Learn the basics of CloudForge AI platform and its core capabilities',
        content: 'Comprehensive introduction covering platform overview, key features, and getting started guide.',
        duration: 15,
        difficulty: 'Beginner',
        type: 'video',
        prerequisites: [],
        tags: ['introduction', 'basics', 'platform'],
        isActive: true,
        createdAt: now,
        updatedAt: now
      }
    ];

    await this.saveData();
  }

  // Create or update training progress for a user
  async createTrainingProgress(userId: string, moduleId: string, status: string): Promise<TrainingProgress> {
    // Verify module exists
    const module = this.storage.modules.find(m => m.id === moduleId);
    if (!module) {
      throw new NotFoundException(`Training module ${moduleId} not found`);
    }

    const now = new Date();
    const progressData: TrainingProgress = {
      userId,
      moduleId,
      status: status as any,
      progress: status === 'completed' ? 100 : 10,
      timeSpent: 0,
      lastAccessed: now,
      createdAt: now,
      updatedAt: now
    };

    this.storage.progress.push(progressData);
    await this.saveData();

    return progressData;
  }

  // Get all available training modules
  async getTrainingModules(): Promise<TrainingModule[]> {
    return this.storage.modules.filter(m => m.isActive);
  }

  // Get user statistics
  async getUserStats(userId: string): Promise<UserStats> {
    let stats = this.storage.stats.find(s => s.userId === userId);
    
    if (!stats) {
      const now = new Date();
      stats = {
        userId,
        totalModules: this.storage.modules.filter(m => m.isActive).length,
        completedModules: 0,
        inProgressModules: 0,
        totalTimeSpent: 0,
        averageScore: 0,
        currentStreak: 0,
        lastLoginDate: now,
        joinedDate: now
      };
      this.storage.stats.push(stats);
      await this.saveData();
    }

    return stats;
  }

  // Update time spent on a module
  async updateTimeSpent(userId: string, moduleId: string, additionalMinutes: number): Promise<void> {
    const progressIndex = this.storage.progress.findIndex(p => 
      p.userId === userId && p.moduleId === moduleId
    );

    if (progressIndex >= 0) {
      this.storage.progress[progressIndex].timeSpent += additionalMinutes;
      this.storage.progress[progressIndex].lastAccessed = new Date();
      this.storage.progress[progressIndex].updatedAt = new Date();
      await this.saveData();
    }
  }

  // Get leaderboard data
  async getLeaderboard(limit: number = 10): Promise<UserStats[]> {
    return this.storage.stats
      .sort((a, b) => {
        // Sort by completed modules first, then by time spent (less is better)
        if (a.completedModules !== b.completedModules) {
          return b.completedModules - a.completedModules;
        }
        return a.totalTimeSpent - b.totalTimeSpent;
      })
      .slice(0, limit);
  }

  async onModuleDestroy() {
    await this.saveData();
  }
}