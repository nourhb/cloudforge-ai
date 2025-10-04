'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ServerIcon, 
  CloudIcon, 
  CircleStackIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useApi } from '@/lib/hooks/useApi';

interface ServiceHealthData {
  services: Array<{
    name: string;
    status: 'healthy' | 'warning' | 'critical' | 'maintenance';
    uptime: number;
    responseTime: number;
    lastCheck: string;
    version: string;
    description: string;
    icon: React.ComponentType<any>;
    metrics: {
      cpu: number;
      memory: number;
      requests: number;
    };
  }>;
  overall: {
    status: 'operational' | 'degraded' | 'down';
    score: number;
  };
}

const ServiceCard: React.FC<{ service: ServiceHealthData['services'][0] }> = ({ service }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 border-green-200 bg-green-50';
      case 'warning': return 'text-yellow-600 border-yellow-200 bg-yellow-50';
      case 'critical': return 'text-red-600 border-red-200 bg-red-50';
      case 'maintenance': return 'text-blue-600 border-blue-200 bg-blue-50';
      default: return 'text-gray-600 border-gray-200 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleIcon className="h-4 w-4" />;
      case 'warning': return <ExclamationTriangleIcon className="h-4 w-4" />;
      case 'critical': return <ExclamationTriangleIcon className="h-4 w-4" />;
      case 'maintenance': return <ClockIcon className="h-4 w-4" />;
      default: return <CheckCircleIcon className="h-4 w-4" />;
    }
  };

  const Icon = service.icon;

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="group"
    >
      <Card className="border-2 border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 transition-all duration-300">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900">
                <Icon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold">{service.name}</CardTitle>
                <CardDescription className="text-sm">v{service.version}</CardDescription>
              </div>
            </div>
            <Badge className={`${getStatusColor(service.status)} border`}>
              <div className="flex items-center space-x-1">
                {getStatusIcon(service.status)}
                <span className="capitalize">{service.status}</span>
              </div>
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Service Description */}
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {service.description}
          </p>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500 dark:text-gray-400">Uptime:</span>
              <p className="font-medium">{service.uptime}%</p>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Response Time:</span>
              <p className="font-medium">{service.responseTime}ms</p>
            </div>
          </div>

          {/* Resource Usage */}
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500 dark:text-gray-400">CPU Usage</span>
                <span className="font-medium">{service.metrics.cpu}%</span>
              </div>
              <Progress value={service.metrics.cpu} className="h-1" />
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500 dark:text-gray-400">Memory Usage</span>
                <span className="font-medium">{service.metrics.memory}%</span>
              </div>
              <Progress value={service.metrics.memory} className="h-1" />
            </div>
          </div>

          {/* Last Check */}
          <div className="text-xs text-gray-500 dark:text-gray-400 pt-2 border-t">
            Last checked: {new Date(service.lastCheck).toLocaleString()}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const OverallHealthCard: React.FC<{ overall: ServiceHealthData['overall'] }> = ({ overall }) => {
  const getOverallColor = (status: string) => {
    switch (status) {
      case 'operational': return 'text-green-600 bg-green-100 border-green-200';
      case 'degraded': return 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'down': return 'text-red-600 bg-red-100 border-red-200';
      default: return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 95) return 'text-green-600';
    if (score >= 85) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card className="border-2 border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <ChartBarIcon className="h-5 w-5" />
          <span>Overall System Health</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between mb-4">
          <Badge className={`${getOverallColor(overall.status)} border`}>
            <CheckCircleIcon className="h-4 w-4 mr-1" />
            <span className="capitalize">{overall.status}</span>
          </Badge>
          <div className="text-right">
            <div className={`text-2xl font-bold ${getScoreColor(overall.score)}`}>
              {overall.score}%
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Health Score
            </div>
          </div>
        </div>
        
        <Progress value={overall.score} className="h-2" />
        
        <p className="text-sm text-gray-600 dark:text-gray-300 mt-3">
          All core services are {overall.status === 'operational' ? 'running smoothly' : 
                                overall.status === 'degraded' ? 'experiencing minor issues' : 
                                'experiencing significant problems'}
        </p>
      </CardContent>
    </Card>
  );
};

export const ServiceHealth: React.FC = () => {
  const [healthData, setHealthData] = useState<ServiceHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const { get } = useApi();

  useEffect(() => {
    const fetchServiceHealth = async () => {
      try {
        setLoading(true);
        // Mock data for demonstration - replace with actual API call
        const mockHealthData: ServiceHealthData = {
          services: [
            {
              name: 'Backend API',
              status: 'healthy',
              uptime: 99.9,
              responseTime: 245,
              lastCheck: new Date().toISOString(),
              version: '2.0.0',
              description: 'Core CloudForge API handling all backend operations',
              icon: ServerIcon,
              metrics: { cpu: 45, memory: 62, requests: 1240 },
            },
            {
              name: 'AI Services',
              status: 'healthy',
              uptime: 99.7,
              responseTime: 180,
              lastCheck: new Date().toISOString(),
              version: '1.5.2',
              description: 'Machine learning and AI processing services',
              icon: CpuChipIcon,
              metrics: { cpu: 78, memory: 84, requests: 890 },
            },
            {
              name: 'Database Cluster',
              status: 'healthy',
              uptime: 100.0,
              responseTime: 35,
              lastCheck: new Date().toISOString(),
              version: '14.2',
              description: 'Primary PostgreSQL database cluster',
              icon: CircleStackIcon,
              metrics: { cpu: 32, memory: 58, requests: 2100 },
            },
            {
              name: 'Authentication',
              status: 'healthy',
              uptime: 99.8,
              responseTime: 120,
              lastCheck: new Date().toISOString(),
              version: '1.8.0',
              description: 'JWT-based authentication and authorization',
              icon: ShieldCheckIcon,
              metrics: { cpu: 25, memory: 41, requests: 340 },
            },
            {
              name: 'Worker Orchestrator',
              status: 'warning',
              uptime: 98.5,
              responseTime: 320,
              lastCheck: new Date().toISOString(),
              version: '2.1.1',
              description: 'Kubernetes job orchestration for marketplace workers',
              icon: CloudIcon,
              metrics: { cpu: 89, memory: 76, requests: 125 },
            },
            {
              name: 'Monitoring Stack',
              status: 'healthy',
              uptime: 99.6,
              responseTime: 95,
              lastCheck: new Date().toISOString(),
              version: '3.0.0',
              description: 'Prometheus, Grafana, and alerting systems',
              icon: ChartBarIcon,
              metrics: { cpu: 55, memory: 68, requests: 450 },
            },
          ],
          overall: {
            status: 'operational',
            score: 98.5,
          },
        };

        setHealthData(mockHealthData);
      } catch (error) {
        console.error('Failed to fetch service health:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchServiceHealth();
    const interval = setInterval(fetchServiceHealth, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [get]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Card className="animate-pulse">
          <CardHeader className="space-y-2">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3" />
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2" />
          </CardHeader>
        </Card>
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
      </div>
    );
  }

  if (!healthData) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500 dark:text-gray-400">
            Failed to load service health data
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall Health Summary */}
      <OverallHealthCard overall={healthData.overall} />

      {/* Individual Service Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {healthData.services.map((service) => (
          <ServiceCard key={service.name} service={service} />
        ))}
      </div>
    </div>
  );
};