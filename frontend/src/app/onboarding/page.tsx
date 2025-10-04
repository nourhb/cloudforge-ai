'use client'

import React, { useState, useEffect } from 'react'
import { 
  CheckCircleIcon, 
  PlayCircleIcon, 
  BookOpenIcon,
  ClockIcon,
  UserGroupIcon,
  AcademicCapIcon,
  RocketLaunchIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

interface TrainingModule {
  id: string
  title: string
  description: string
  duration: number // in minutes
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  status: 'not_started' | 'in_progress' | 'completed'
  progress: number // 0-100
  type: 'video' | 'interactive' | 'reading' | 'hands-on'
}

interface UserProgress {
  userId: string
  totalModules: number
  completedModules: number
  totalTime: number
  lastAccessed: string
  currentStreak: number
}

export default function OnboardingPage() {
  const [modules, setModules] = useState<TrainingModule[]>([])
  const [userProgress, setUserProgress] = useState<UserProgress | null>(null)
  const [selectedModule, setSelectedModule] = useState<TrainingModule | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Sample training modules data
  const sampleModules: TrainingModule[] = [
    {
      id: 'intro-cloudforge',
      title: 'Introduction to CloudForge AI',
      description: 'Learn the basics of CloudForge AI platform and its core capabilities',
      duration: 15,
      difficulty: 'Beginner',
      status: 'completed',
      progress: 100,
      type: 'video'
    },
    {
      id: 'infrastructure-basics',
      title: 'Infrastructure as Code Fundamentals',
      description: 'Understanding IaC concepts and how CloudForge AI automates infrastructure',
      duration: 25,
      difficulty: 'Beginner',
      status: 'in_progress',
      progress: 60,
      type: 'interactive'
    },
    {
      id: 'database-migration',
      title: 'AI-Powered Database Migration',
      description: 'Master database migration using AI-driven analysis and automation',
      duration: 45,
      difficulty: 'Intermediate',
      status: 'not_started',
      progress: 0,
      type: 'hands-on'
    },
    {
      id: 'worker-marketplace',
      title: 'Worker Marketplace & Orchestration',
      description: 'Deploy and manage custom applications in the worker marketplace',
      duration: 30,
      difficulty: 'Intermediate',
      status: 'not_started',
      progress: 0,
      type: 'hands-on'
    },
    {
      id: 'monitoring-analytics',
      title: 'Monitoring & Analytics Dashboard',
      description: 'Set up comprehensive monitoring with real-time analytics',
      duration: 20,
      difficulty: 'Intermediate',
      status: 'not_started',
      progress: 0,
      type: 'reading'
    },
    {
      id: 'advanced-security',
      title: 'Enterprise Security & Compliance',
      description: 'Implement advanced security features and compliance standards',
      duration: 35,
      difficulty: 'Advanced',
      status: 'not_started',
      progress: 0,
      type: 'video'
    }
  ]

  // Mock user progress data
  const mockUserProgress: UserProgress = {
    userId: 'user-123',
    totalModules: 6,
    completedModules: 1,
    totalTime: 75, // minutes spent
    lastAccessed: new Date().toISOString(),
    currentStreak: 3
  }

  useEffect(() => {
    // Simulate API call to fetch user progress and modules
    setTimeout(() => {
      setModules(sampleModules)
      setUserProgress(mockUserProgress)
      setIsLoading(false)
    }, 1000)
  }, [])

  const startModule = async (moduleId: string) => {
    try {
      // Update module status to in_progress
      setModules(prev => 
        prev.map(module => 
          module.id === moduleId 
            ? { ...module, status: 'in_progress' as const, progress: 10 }
            : module
        )
      )

      // API call to update progress in backend
      const response = await fetch('/api/onboarding/progress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userProgress?.userId,
          moduleId,
          status: 'in_progress',
          progress: 10
        })
      })

      if (!response.ok) {
        throw new Error('Failed to update progress')
      }

      console.log(`Started module: ${moduleId}`)
    } catch (error) {
      console.error('Error starting module:', error)
    }
  }

  const completeModule = async (moduleId: string) => {
    try {
      // Update module status to completed
      setModules(prev => 
        prev.map(module => 
          module.id === moduleId 
            ? { ...module, status: 'completed' as const, progress: 100 }
            : module
        )
      )

      // Update user progress
      if (userProgress) {
        setUserProgress(prev => prev ? {
          ...prev,
          completedModules: prev.completedModules + 1
        } : null)
      }

      // API call to update progress in backend
      const response = await fetch('/api/onboarding/progress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userProgress?.userId,
          moduleId,
          status: 'completed',
          progress: 100
        })
      })

      if (!response.ok) {
        throw new Error('Failed to update progress')
      }

      console.log(`Completed module: ${moduleId}`)
    } catch (error) {
      console.error('Error completing module:', error)
    }
  }

  const getModuleIcon = (type: string) => {
    switch (type) {
      case 'video': return <PlayCircleIcon className="h-6 w-6" />
      case 'reading': return <BookOpenIcon className="h-6 w-6" />
      case 'interactive': return <AcademicCapIcon className="h-6 w-6" />
      case 'hands-on': return <RocketLaunchIcon className="h-6 w-6" />
      default: return <BookOpenIcon className="h-6 w-6" />
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800'
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800'  
      case 'Advanced': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading your training modules...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              CloudForge AI Training Center
            </h1>
            <p className="mt-2 text-gray-600">
              Master the CloudForge AI platform with our comprehensive training modules
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Progress Overview */}
        {userProgress && (
          <div className="bg-white rounded-lg shadow mb-8 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Your Progress</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {userProgress.completedModules}/{userProgress.totalModules}
                </div>
                <div className="text-sm text-gray-600">Modules Completed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {Math.round((userProgress.completedModules / userProgress.totalModules) * 100)}%
                </div>
                <div className="text-sm text-gray-600">Overall Progress</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {userProgress.totalTime}m
                </div>
                <div className="text-sm text-gray-600">Time Invested</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {userProgress.currentStreak}
                </div>
                <div className="text-sm text-gray-600">Day Streak</div>
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-4">
              <div className="bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  data-progress={(userProgress.completedModules / userProgress.totalModules) * 100}
                ></div>
              </div>
            </div>
          </div>
        )}

        {/* Training Modules Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module) => (
            <div key={module.id} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
              {/* Module Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-2">
                  {getModuleIcon(module.type)}
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(module.difficulty)}`}>
                    {module.difficulty}
                  </span>
                </div>
                {module.status === 'completed' && (
                  <CheckCircleIcon className="h-6 w-6 text-green-500" />
                )}
              </div>

              {/* Module Content */}
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {module.title}
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                {module.description}
              </p>

              {/* Module Metadata */}
              <div className="flex items-center text-sm text-gray-500 mb-4">
                <ClockIcon className="h-4 w-4 mr-1" />
                <span>{module.duration} minutes</span>
                <span className="mx-2">•</span>
                <span className="capitalize">{module.type}</span>
              </div>

              {/* Progress Bar */}
              {module.status !== 'not_started' && (
                <div className="mb-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Progress</span>
                    <span>{module.progress}%</span>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${
                        module.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                      }`}
                      data-progress={module.progress}
                    ></div>
                  </div>
                </div>
              )}

              {/* Action Button */}
              <div className="flex justify-between items-center">
                {module.status === 'not_started' && (
                  <button
                    onClick={() => startModule(module.id)}
                    className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors flex items-center space-x-2"
                  >
                    <PlayCircleIcon className="h-4 w-4" />
                    <span>Start Module</span>
                  </button>
                )}
                {module.status === 'in_progress' && (
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setSelectedModule(module)}
                      className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors"
                    >
                      Continue
                    </button>
                    <button
                      onClick={() => completeModule(module.id)}
                      className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
                    >
                      Mark Complete
                    </button>
                  </div>
                )}
                {module.status === 'completed' && (
                  <div className="flex items-center space-x-2 text-green-600">
                    <CheckCircleIcon className="h-5 w-5" />
                    <span className="font-medium">Completed</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Learning Path Recommendations */}
        <div className="mt-12 bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <ChartBarIcon className="h-6 w-6 mr-2" />
            Recommended Learning Path
          </h2>
          <div className="space-y-3">
            <div className="flex items-center text-sm">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                <span className="text-blue-600 font-semibold">1</span>
              </div>
              <span>Start with CloudForge AI basics to understand core concepts</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                <span className="text-blue-600 font-semibold">2</span>
              </div>
              <span>Learn Infrastructure as Code fundamentals</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                <span className="text-blue-600 font-semibold">3</span>
              </div>
              <span>Practice with hands-on database migration and worker marketplace</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                <span className="text-blue-600 font-semibold">4</span>
              </div>
              <span>Set up monitoring and implement advanced security features</span>
            </div>
          </div>
        </div>
      </div>

      {/* Module Detail Modal */}
      {selectedModule && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-semibold">{selectedModule.title}</h3>
                <button
                  onClick={() => setSelectedModule(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <p className="text-gray-600 mb-4">{selectedModule.description}</p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Module Content:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Interactive exercises and examples</li>
                  <li>• Step-by-step guides and tutorials</li>
                  <li>• Real-world scenarios and use cases</li>
                  <li>• Hands-on practice environment</li>
                  <li>• Knowledge check quizzes</li>
                </ul>
              </div>
              <div className="mt-6 flex justify-end space-x-3">
                <button
                  onClick={() => setSelectedModule(null)}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    completeModule(selectedModule.id)
                    setSelectedModule(null)
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Mark as Complete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}