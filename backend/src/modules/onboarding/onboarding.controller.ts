import { Controller, Get, Post, Body, Param, Query } from '@nestjs/common';
import { OnboardingService, TrainingProgress, UserStats, TrainingModule } from './onboarding.service';

export interface CreateProgressDto {
  userId: string;
  moduleId: string;
  status: 'not_started' | 'in_progress' | 'completed';
  progress?: number;
}

export interface UpdateTimeDto {
  userId: string;
  moduleId: string;
  additionalMinutes: number;
}

@Controller('api/onboarding')
export class OnboardingController {
  constructor(private readonly onboardingService: OnboardingService) {}

  // Get all available training modules
  @Get('modules')
  async getTrainingModules(): Promise<TrainingModule[]> {
    return this.onboardingService.getTrainingModules();
  }

  // Create or update user progress
  @Post('progress')
  async createTrainingProgress(@Body() createProgressDto: CreateProgressDto): Promise<TrainingProgress> {
    const { userId, moduleId, status } = createProgressDto;
    return this.onboardingService.createTrainingProgress(userId, moduleId, status);
  }

  // Get user statistics
  @Get('stats/:userId')
  async getUserStats(@Param('userId') userId: string): Promise<UserStats> {
    return this.onboardingService.getUserStats(userId);
  }

  // Update time spent on a module
  @Post('time')
  async updateTimeSpent(@Body() updateTimeDto: UpdateTimeDto): Promise<{ success: boolean }> {
    const { userId, moduleId, additionalMinutes } = updateTimeDto;
    await this.onboardingService.updateTimeSpent(userId, moduleId, additionalMinutes);
    return { success: true };
  }

  // Get leaderboard
  @Get('leaderboard')
  async getLeaderboard(@Query('limit') limit?: string): Promise<UserStats[]> {
    const limitNum = limit ? parseInt(limit, 10) : 10;
    return this.onboardingService.getLeaderboard(limitNum);
  }
}