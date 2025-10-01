'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  SparklesIcon, 
  PlayIcon, 
  PauseIcon,
  StopIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useApi } from '@/lib/hooks/useApi';

interface AIWorkflow {
  id: string;
  name: string;
  type: 'migration' | 'iac' | 'monitoring' | 'optimization';
  status: 'running' | 'completed' | 'paused' | 'failed' | 'queued';
  progress: number;
  startTime: string;
  estimatedCompletion?: string;
  description: string;
  logs: Array<{
    timestamp: string;
    level: 'info' | 'warning' | 'error' | 'success';
    message: string;
  }>;
}

const WorkflowCard: React.FC<{ workflow: AIWorkflow; onAction: (id: string, action: string) => void }> = ({ 
  workflow, 
  onAction 
}) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'completed': return 'bg-green-100 text-green-800 border-green-200';
      case 'paused': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'failed': return 'bg-red-100 text-red-800 border-red-200';
      case 'queued': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <PlayIcon className="h-4 w-4" />;
      case 'completed': return <CheckCircleIcon className="h-4 w-4" />;
      case 'paused': return <PauseIcon className="h-4 w-4" />;
      case 'failed': return <ExclamationTriangleIcon className="h-4 w-4" />;
      case 'queued': return <ClockIcon className="h-4 w-4" />;
      default: return <InformationCircleIcon className="h-4 w-4" />;
    }
  };

  const canPause = workflow.status === 'running';
  const canResume = workflow.status === 'paused';
  const canStop = workflow.status === 'running' || workflow.status === 'paused';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="group"
    >
      <Card className="border-2 border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 transition-all duration-300">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">{workflow.name}</CardTitle>
            <Badge className={`${getStatusColor(workflow.status)} border`}>
              <div className="flex items-center space-x-1">
                {getStatusIcon(workflow.status)}
                <span className="capitalize">{workflow.status}</span>
              </div>
            </Badge>
          </div>
          <CardDescription className="text-sm text-gray-500 dark:text-gray-400">
            {workflow.description}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-300">Progress</span>
              <span className="font-medium">{workflow.progress}%</span>
            </div>
            <Progress value={workflow.progress} className="h-2" />
          </div>

          {/* Timing Information */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500 dark:text-gray-400">Started:</span>
              <p className="font-medium">{new Date(workflow.startTime).toLocaleTimeString()}</p>
            </div>
            {workflow.estimatedCompletion && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">ETA:</span>
                <p className="font-medium">{new Date(workflow.estimatedCompletion).toLocaleTimeString()}</p>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-2 pt-2">
            {canPause && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAction(workflow.id, 'pause')}
                className="flex items-center space-x-1"
              >
                <PauseIcon className="h-3 w-3" />
                <span>Pause</span>
              </Button>
            )}
            {canResume && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAction(workflow.id, 'resume')}
                className="flex items-center space-x-1"
              >
                <PlayIcon className="h-3 w-3" />
                <span>Resume</span>
              </Button>
            )}
            {canStop && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAction(workflow.id, 'stop')}
                className="flex items-center space-x-1 hover:bg-red-50 hover:border-red-200"
              >
                <StopIcon className="h-3 w-3" />
                <span>Stop</span>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const LogViewer: React.FC<{ logs: AIWorkflow['logs'] }> = ({ logs }) => {
  const getLogColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-600 dark:text-red-400';
      case 'warning': return 'text-yellow-600 dark:text-yellow-400';
      case 'success': return 'text-green-600 dark:text-green-400';
      default: return 'text-gray-600 dark:text-gray-300';
    }
  };

  return (
    <div className="max-h-64 overflow-y-auto space-y-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
      {logs.map((log, index) => (
        <div key={index} className="text-xs font-mono">
          <span className="text-gray-500 dark:text-gray-400">
            {new Date(log.timestamp).toLocaleTimeString()}
          </span>
          <span className={`ml-2 ${getLogColor(log.level)}`}>
            [{log.level.toUpperCase()}]
          </span>
          <span className="ml-2 text-gray-700 dark:text-gray-200">
            {log.message}
          </span>
        </div>
      ))}
      {logs.length === 0 && (
        <p className="text-center text-gray-500 dark:text-gray-400 py-4">
          No logs available
        </p>
      )}
    </div>
  );
};

export const AIWorkflowPanel: React.FC = () => {
  const [workflows, setWorkflows] = useState<AIWorkflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const { get, post } = useApi();

  useEffect(() => {
    const fetchWorkflows = async () => {
      try {
        setLoading(true);
        // Mock data for demonstration - replace with actual API call
        const mockWorkflows: AIWorkflow[] = [
          {
            id: '1',
            name: 'PostgreSQL to K8s Migration',
            type: 'migration',
            status: 'running',
            progress: 67,
            startTime: new Date(Date.now() - 3600000).toISOString(),
            estimatedCompletion: new Date(Date.now() + 1800000).toISOString(),
            description: 'Migrating customer database to Kubernetes with AI optimization',
            logs: [
              {
                timestamp: new Date(Date.now() - 300000).toISOString(),
                level: 'info',
                message: 'Schema analysis completed successfully',
              },
              {
                timestamp: new Date(Date.now() - 180000).toISOString(),
                level: 'success',
                message: 'Data validation passed for 15/20 tables',
              },
              {
                timestamp: new Date(Date.now() - 60000).toISOString(),
                level: 'warning',
                message: 'Large table detected, switching to batch processing',
              },
            ],
          },
          {
            id: '2',
            name: 'Terraform Config Generation',
            type: 'iac',
            status: 'completed',
            progress: 100,
            startTime: new Date(Date.now() - 1800000).toISOString(),
            description: 'Generated AWS infrastructure configuration from natural language',
            logs: [
              {
                timestamp: new Date(Date.now() - 900000).toISOString(),
                level: 'info',
                message: 'Parsing natural language requirements',
              },
              {
                timestamp: new Date(Date.now() - 600000).toISOString(),
                level: 'success',
                message: 'Terraform configuration generated successfully',
              },
            ],
          },
          {
            id: '3',
            name: 'Performance Optimization',
            type: 'optimization',
            status: 'queued',
            progress: 0,
            startTime: new Date().toISOString(),
            description: 'AI-driven resource optimization for microservices cluster',
            logs: [],
          },
        ];

        setWorkflows(mockWorkflows);
      } catch (error) {
        console.error('Failed to fetch workflows:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflows();
    const interval = setInterval(fetchWorkflows, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [get]);

  const handleWorkflowAction = async (workflowId: string, action: string) => {
    try {
      await post(`/api/workflows/${workflowId}/action`, { action });
      // Refresh workflows after action
      // In real implementation, you would refetch the workflows
    } catch (error) {
      console.error(`Failed to ${action} workflow:`, error);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Workflows</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 dark:bg-gray-700 rounded" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const runningWorkflows = workflows.filter(w => w.status === 'running');
  const completedWorkflows = workflows.filter(w => w.status === 'completed');
  const otherWorkflows = workflows.filter(w => !['running', 'completed'].includes(w.status));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <SparklesIcon className="h-5 w-5" />
          <span>AI Workflows</span>
        </CardTitle>
        <CardDescription>
          Monitor and control your automated AI-powered processes
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="active" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="active">
              Active ({runningWorkflows.length})
            </TabsTrigger>
            <TabsTrigger value="completed">
              Completed ({completedWorkflows.length})
            </TabsTrigger>
            <TabsTrigger value="other">
              Other ({otherWorkflows.length})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="active" className="space-y-4 mt-6">
            {runningWorkflows.length > 0 ? (
              runningWorkflows.map(workflow => (
                <WorkflowCard
                  key={workflow.id}
                  workflow={workflow}
                  onAction={handleWorkflowAction}
                />
              ))
            ) : (
              <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                No active workflows
              </p>
            )}
          </TabsContent>
          
          <TabsContent value="completed" className="space-y-4 mt-6">
            {completedWorkflows.length > 0 ? (
              completedWorkflows.map(workflow => (
                <WorkflowCard
                  key={workflow.id}
                  workflow={workflow}
                  onAction={handleWorkflowAction}
                />
              ))
            ) : (
              <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                No completed workflows
              </p>
            )}
          </TabsContent>
          
          <TabsContent value="other" className="space-y-4 mt-6">
            {otherWorkflows.length > 0 ? (
              otherWorkflows.map(workflow => (
                <WorkflowCard
                  key={workflow.id}
                  workflow={workflow}
                  onAction={handleWorkflowAction}
                />
              ))
            ) : (
              <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                No other workflows
              </p>
            )}
          </TabsContent>
        </Tabs>

        {/* Detailed Log View */}
        {selectedWorkflow && (
          <div className="mt-6 p-4 border rounded-lg">
            <h4 className="font-semibold mb-3">Workflow Logs</h4>
            <LogViewer 
              logs={workflows.find(w => w.id === selectedWorkflow)?.logs || []} 
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};