import { Injectable, NestMiddleware, HttpStatus } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import { BillingService } from '../modules/billing/billing.service';

// Enhanced request interface to include user info
interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

// Rate limiting configuration for different tiers
export const freeTierLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour window
  max: 10, // Limit each IP/user to 10 requests per windowMs
  message: {
    error: 'Rate limit exceeded',
    message: 'Free tier limit reached (10 requests/hour). Upgrade to Pro plan for higher limits.',
    type: 'RATE_LIMIT_EXCEEDED',
    upgradeUrl: '/billing/upgrade',
    resetTime: new Date(Date.now() + 60 * 60 * 1000)
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
  keyGenerator: (req: AuthenticatedRequest) => {
    // Use user ID if authenticated, otherwise fall back to IP
    return req.user?.id || req.ip;
  },
  handler: (req: AuthenticatedRequest, res: Response) => {
    const resetTime = new Date(Date.now() + 60 * 60 * 1000);
    res.status(HttpStatus.TOO_MANY_REQUESTS).json({
      error: 'Rate limit exceeded',
      message: 'Free tier limit reached (10 requests/hour). Upgrade to Pro plan for higher limits.',
      type: 'RATE_LIMIT_EXCEEDED',
      upgradeUrl: '/api/billing/plans',
      resetTime,
      currentPlan: 'free',
      limits: {
        requestsPerHour: 10,
        upgradeOptions: {
          pro: { requestsPerHour: 1000, price: 49 },
          enterprise: { requestsPerHour: 10000, price: 199 }
        }
      }
    });
  }
});

export const proTierLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour window
  max: 1000, // Pro tier: 1000 requests per hour
  message: {
    error: 'Rate limit exceeded',
    message: 'Pro tier limit reached (1000 requests/hour). Contact support for enterprise options.',
    type: 'RATE_LIMIT_EXCEEDED',
    contactUrl: '/support'
  },
  keyGenerator: (req: AuthenticatedRequest) => {
    return req.user?.id || req.ip;
  }
});

export const enterpriseTierLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour window
  max: 10000, // Enterprise tier: 10000 requests per hour
  message: {
    error: 'Rate limit exceeded',
    message: 'Enterprise tier limit reached (10000 requests/hour). Contact your account manager.',
    type: 'RATE_LIMIT_EXCEEDED'
  },
  keyGenerator: (req: AuthenticatedRequest) => {
    return req.user?.id || req.ip;
  }
});

// Custom rate limiting middleware that checks user's billing plan
@Injectable()
export class DynamicRateLimitMiddleware implements NestMiddleware {
  constructor(private readonly billingService: BillingService) {}

  async use(req: AuthenticatedRequest, res: Response, next: NextFunction) {
    // Skip rate limiting for health checks and static assets
    if (req.path.includes('/health') || req.path.includes('/static')) {
      return next();
    }

    try {
      // Get user ID from authenticated request
      const userId = req.user?.id;
      
      if (!userId) {
        // Apply free tier limits for unauthenticated requests
        return freeTierLimiter(req, res, next);
      }

      // Check user's billing plan and apply appropriate rate limiting
      const rateLimitResult = await this.billingService.canMakeRequest(userId);
      
      if (!rateLimitResult.allowed) {
        return res.status(HttpStatus.TOO_MANY_REQUESTS).json({
          error: 'Rate limit exceeded',
          message: 'Request limit reached for your current plan. Upgrade for higher limits.',
          type: 'RATE_LIMIT_EXCEEDED',
          resetAt: rateLimitResult.resetAt,
          remaining: rateLimitResult.remaining,
          upgradeUrl: '/api/billing/plans'
        });
      }

      // Add rate limit headers
      res.set({
        'X-RateLimit-Remaining': rateLimitResult.remaining?.toString() || '0',
        'X-RateLimit-Reset': rateLimitResult.resetAt?.getTime().toString() || '0'
      });

      next();
    } catch (error) {
      console.error('Rate limiting error:', error);
      // Fall back to free tier limits on error
      return freeTierLimiter(req, res, next);
    }
  }
}

// API-specific rate limiting for different endpoints
export const apiRateLimits = {
  // AI endpoints (more restrictive)
  aiGeneration: rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5, // Free tier: 5 AI generations per hour
    message: {
      error: 'AI generation rate limit exceeded',
      message: 'Free tier allows 5 AI generations per hour. Upgrade to Pro for 100/hour.',
      upgradeUrl: '/api/billing/plans'
    }
  }),

  // File uploads (moderate restriction)
  fileUpload: rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 file uploads per 15 minutes
    message: {
      error: 'File upload rate limit exceeded',
      message: 'Too many file uploads. Please wait before uploading more files.'
    }
  }),

  // Database operations (less restrictive)
  database: rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 100, // 100 database operations per hour
    message: {
      error: 'Database operation rate limit exceeded',
      message: 'Database operation limit reached. Upgrade for higher limits.'
    }
  }),

  // Authentication endpoints
  auth: rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 login attempts per 15 minutes
    message: {
      error: 'Authentication rate limit exceeded',
      message: 'Too many login attempts. Please wait 15 minutes before trying again.'
    }
  })
};

// Kong API Gateway rate limiting configuration (for production)
export const kongRateLimitConfig = {
  free: {
    minute: 2,
    hour: 10,
    day: 50
  },
  pro: {
    minute: 100,
    hour: 1000,
    day: 10000
  },
  enterprise: {
    minute: 1000,
    hour: 10000,
    day: 100000
  }
};

// Rate limit bypass for admin users
export const adminBypassMiddleware = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  if (req.user?.role === 'admin') {
    // Skip rate limiting for admin users
    return next();
  }
  
  // Continue with normal rate limiting
  next();
};

// Custom rate limit storage (Redis in production)
class CustomRateLimitStore {
  private hits: Map<string, { count: number; resetTime: number }> = new Map();

  async increment(key: string): Promise<{ totalHits: number; timeToExpire: number }> {
    const now = Date.now();
    const windowMs = 60 * 60 * 1000; // 1 hour
    const resetTime = now + windowMs;

    const existing = this.hits.get(key);
    
    if (!existing || now > existing.resetTime) {
      // New window or expired
      this.hits.set(key, { count: 1, resetTime });
      return { totalHits: 1, timeToExpire: windowMs };
    }

    // Increment existing
    existing.count++;
    return { 
      totalHits: existing.count, 
      timeToExpire: existing.resetTime - now 
    };
  }

  async decrement(key: string): Promise<void> {
    const existing = this.hits.get(key);
    if (existing && existing.count > 0) {
      existing.count--;
    }
  }

  async resetKey(key: string): Promise<void> {
    this.hits.delete(key);
  }
}

// Export custom store for advanced use cases
export const customRateLimitStore = new CustomRateLimitStore();