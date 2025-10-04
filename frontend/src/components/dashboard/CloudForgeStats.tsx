'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  CloudIcon, 
  CpuChipIcon, 
  ServerIcon,
  CircleStackIcon,
  CodeBracketIcon
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
// Update the import path below to the correct relative path if needed
import { useApi } from '../../lib/hooks/useApi';

interface CloudForgeStatsData {
  totalMigrations: number;
  activeMigrations: number;
  totalWorkers: number;
  activeWorkers: number;
  iacTemplates: number;
  generatedConfigs: number;
  systemHealth: {
    backend: 'healthy' | 'warning' | 'error';
    aiServices: 'healthy' | 'warning' | 'error';
    database: 'healthy' | 'warning' | 'error';
  };
  recentActivity: Array<{
    id: string;
    type: 'migration' | 'worker' | 'iac' | 'deployment';
    description: string;
    timestamp: string;
    status: 'success' | 'pending' | 'error';
  }>;
}

const StatCard: React.FC<{
  icon: React.ComponentType<any>;
  title: string;
  value: number;
  change: number;
  color: string;
  description: string;
}> = ({ icon: Icon, title, value, change, color, description }) => (
  <motion.div
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    className="group"
  >
    <Card className="h-full border-2 border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 transition-all duration-300">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-300">
          {title}
        </CardTitle>
        <div className={`p-2 rounded-lg ${color} bg-opacity-10`}>
          <Icon className={`h-4 w-4 ${color.replace('bg-', 'text-')}`} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
          {value.toLocaleString()}
        </div>
        <div className="flex items-center text-xs">
          <span className={`font-medium ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {change >= 0 ? '+' : ''}{change}%
          </span>
          <span className="text-gray-500 dark:text-gray-400 ml-1">
            from last month
          </span>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
          {description}
        </p>
      </CardContent>
    </Card>
  </motion.div>
);

const HealthIndicator: React.FC<{
  service: string;
  status: 'healthy' | 'warning' | 'error';
}> = ({ service, status }) => {
  const statusConfig = {
    healthy: { color: 'bg-green-500', text: 'Healthy', textColor: 'text-green-700' },
    warning: { color: 'bg-yellow-500', text: 'Warning', textColor: 'text-yellow-700' },
    error: { color: 'bg-red-500', text: 'Error', textColor: 'text-red-700' },
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
      <span className="text-sm font-medium text-gray-900 dark:text-white">
        {service}
      </span>
      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${config.color}`} />
        <Badge variant="outline" className={config.textColor}>
          {config.text}
        </Badge>
      </div>
    </div>
  );
};

export const CloudForgeStats: React.FC = () => {
  const [stats, setStats] = useState<CloudForgeStatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const { get } = useApi();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        // Simulate API calls to our backend services
        const [migrationsResponse, workersResponse, iacResponse, healthResponse] = await Promise.allSettled([
          get('/api/migration/stats'),
          get('/api/marketplace/stats'),
          get('/api/iac/stats'),
          get('/api/health'),
        ]);

        // Mock data for now - replace with actual API responses
        const mockStats: CloudForgeStatsData = {
          totalMigrations: 247,
          activeMigrations: 12,
          totalWorkers: 89,
          activeWorkers: 23,
          iacTemplates: 156,
          generatedConfigs: 1342,
          systemHealth: {
            backend: 'healthy',
            aiServices: 'healthy',
            database: 'healthy',
          },
          recentActivity: [
            {
              id: '1',
              type: 'migration',
              description: 'PostgreSQL cluster migration completed',
              timestamp: new Date(Date.now() - 300000).toISOString(),
              status: 'success',
            },
            {
              id: '2',
              type: 'iac',
              description: 'Kubernetes deployment manifest generated',
              timestamp: new Date(Date.now() - 600000).toISOString(),
              status: 'success',
            },
            {
              id: '3',
              type: 'worker',
              description: 'New AI script deployed to marketplace',
              timestamp: new Date(Date.now() - 900000).toISOString(),
              status: 'pending',
            },
          ],
        };

        setStats(mockStats);
      } catch (error) {
        console.error('Failed to fetch CloudForge stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [get]);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="space-y-2">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
              <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2" />
            </CardHeader>
          </Card>
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500 dark:text-gray-400">
            Failed to load CloudForge statistics
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <StatCard
          icon={CircleStackIcon}
          title="Database Migrations"
          value={stats.totalMigrations}
          change={15.2}
          color="bg-blue-500"
          description={`${stats.activeMigrations} currently in progress`}
        />
        <StatCard
          icon={ServerIcon}
          title="Active Workers"
          value={stats.activeWorkers}
          change={8.7}
          color="bg-green-500"
          description={`${stats.totalWorkers} total workers deployed`}
        />
        <StatCard
          icon={CodeBracketIcon}
          title="IaC Templates"
          value={stats.iacTemplates}
          change={23.4}
          color="bg-purple-500"
          description={`${stats.generatedConfigs} configs generated`}
        />
      </div>

      {/* System Health and Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <ChartBarIcon className="h-5 w-5" />
              <span>System Health</span>
            </CardTitle>
            <CardDescription>
              Real-time status of CloudForge AI services
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <HealthIndicator service="Backend API" status={stats.systemHealth.backend} />
            <HealthIndicator service="AI Services" status={stats.systemHealth.aiServices} />
            <HealthIndicator service="Database" status={stats.systemHealth.database} />
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CloudIcon className="h-5 w-5" />
              <span>Recent Activity</span>
            </CardTitle>
            <CardDescription>
              Latest operations and deployments
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {stats.recentActivity.map((activity) => (
              <div
                key={activity.id}
                className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800"
              >
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {activity.description}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(activity.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <Badge
                  variant={activity.status === 'success' ? 'default' : 
                          activity.status === 'pending' ? 'secondary' : 'outline'}
                >
                  {activity.status}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};