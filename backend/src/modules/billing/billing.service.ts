import { Injectable, BadRequestException, NotFoundException } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export interface BillingPlan {
  id: string;
  name: string;
  price: number; // USD per month
  currency: string;
  features: string[];
  limits: {
    requestsPerHour: number;
    maxProjects: number;
    maxStorageGB: number;
    maxUsers: number;
    supportLevel: 'community' | 'email' | 'priority';
  };
  isActive: boolean;
}

export interface UserSubscription {
  userId: string;
  planId: string;
  status: 'active' | 'canceled' | 'past_due' | 'trialing';
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
  stripeCustomerId?: string;
  stripeSubscriptionId?: string;
  trialEnd?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface UsageRecord {
  userId: string;
  planId: string;
  requests: number;
  storageUsedGB: number;
  timestamp: Date;
  resetAt: Date;
}

export interface BillingEvent {
  id: string;
  userId: string;
  type: 'subscription_created' | 'subscription_updated' | 'subscription_canceled' | 'payment_succeeded' | 'payment_failed';
  data: any;
  timestamp: Date;
}

interface BillingStorage {
  plans: BillingPlan[];
  subscriptions: UserSubscription[];
  usage: UsageRecord[];
  events: BillingEvent[];
}

@Injectable()
export class BillingService {
  private dataPath: string;
  private storage: BillingStorage = {
    plans: [],
    subscriptions: [],
    usage: [],
    events: []
  };

  constructor() {
    this.dataPath = path.join(process.cwd(), 'data', 'billing-data.json');
    this.initializeBilling();
  }

  private async initializeBilling() {
    // Ensure data directory exists
    const dataDir = path.dirname(this.dataPath);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Load existing data or seed initial plans
    if (fs.existsSync(this.dataPath)) {
      try {
        const rawData = fs.readFileSync(this.dataPath, 'utf8');
        this.storage = JSON.parse(rawData);
        // Convert date strings back to Date objects
        this.storage.subscriptions.forEach(sub => {
          sub.currentPeriodStart = new Date(sub.currentPeriodStart);
          sub.currentPeriodEnd = new Date(sub.currentPeriodEnd);
          sub.createdAt = new Date(sub.createdAt);
          sub.updatedAt = new Date(sub.updatedAt);
          if (sub.trialEnd) sub.trialEnd = new Date(sub.trialEnd);
        });
        this.storage.usage.forEach(usage => {
          usage.timestamp = new Date(usage.timestamp);
          usage.resetAt = new Date(usage.resetAt);
        });
        this.storage.events.forEach(event => {
          event.timestamp = new Date(event.timestamp);
        });
      } catch (error) {
        console.error('Error loading billing data:', error);
        await this.seedBillingPlans();
      }
    } else {
      await this.seedBillingPlans();
    }
  }

  private async saveData() {
    try {
      fs.writeFileSync(this.dataPath, JSON.stringify(this.storage, null, 2));
    } catch (error) {
      console.error('Error saving billing data:', error);
    }
  }

  private async seedBillingPlans() {
    this.storage.plans = [
      {
        id: 'free',
        name: 'Free Tier',
        price: 0,
        currency: 'USD',
        features: [
          'Basic AI-powered database migration',
          'Community support',
          'Basic monitoring dashboard',
          'Limited IaC generation',
          '1 project',
          'Documentation access'
        ],
        limits: {
          requestsPerHour: 10,
          maxProjects: 1,
          maxStorageGB: 1,
          maxUsers: 1,
          supportLevel: 'community'
        },
        isActive: true
      },
      {
        id: 'pro',
        name: 'Pro Plan',
        price: 49,
        currency: 'USD',
        features: [
          'Advanced AI migration analysis',
          'Priority email support',
          'Advanced monitoring & analytics',
          'Unlimited IaC generation',
          'Custom worker marketplace',
          'Advanced security features',
          'Backup & disaster recovery',
          'API access',
          'Custom integrations',
          'Advanced reporting'
        ],
        limits: {
          requestsPerHour: 1000,
          maxProjects: 10,
          maxStorageGB: 100,
          maxUsers: 5,
          supportLevel: 'email'
        },
        isActive: true
      },
      {
        id: 'enterprise',
        name: 'Enterprise Plan',
        price: 199,
        currency: 'USD',
        features: [
          'Everything in Pro',
          'Dedicated support engineer',
          'Custom AI model training',
          'Advanced compliance features',
          'Single sign-on (SSO)',
          'Advanced audit trails',
          'Custom deployment options',
          'SLA guarantees',
          'Phone support',
          'Custom training sessions'
        ],
        limits: {
          requestsPerHour: 10000,
          maxProjects: -1, // Unlimited
          maxStorageGB: 1000,
          maxUsers: -1, // Unlimited
          supportLevel: 'priority'
        },
        isActive: true
      }
    ];

    await this.saveData();
  }

  // Get all available billing plans
  async getBillingPlans(): Promise<BillingPlan[]> {
    return this.storage.plans.filter(plan => plan.isActive);
  }

  // Get specific billing plan
  async getBillingPlan(planId: string): Promise<BillingPlan> {
    const plan = this.storage.plans.find(p => p.id === planId && p.isActive);
    if (!plan) {
      throw new NotFoundException(`Billing plan ${planId} not found`);
    }
    return plan;
  }

  // Subscribe user to a plan
  async subscribeUser(userId: string, planId: string, paymentMethodId?: string): Promise<UserSubscription> {
    const plan = await this.getBillingPlan(planId);
    
    // Check if user already has an active subscription
    const existingSubscription = this.storage.subscriptions.find(
      sub => sub.userId === userId && sub.status === 'active'
    );

    if (existingSubscription) {
      throw new BadRequestException('User already has an active subscription');
    }

    const now = new Date();
    const currentPeriodEnd = new Date();
    currentPeriodEnd.setMonth(currentPeriodEnd.getMonth() + 1);

    const subscription: UserSubscription = {
      userId,
      planId,
      status: plan.price === 0 ? 'active' : 'trialing',
      currentPeriodStart: now,
      currentPeriodEnd,
      cancelAtPeriodEnd: false,
      trialEnd: plan.price > 0 ? new Date(Date.now() + 14 * 24 * 60 * 60 * 1000) : undefined, // 14-day trial
      createdAt: now,
      updatedAt: now
    };

    // Mock Stripe integration for paid plans
    if (plan.price > 0) {
      subscription.stripeCustomerId = `cus_mock_${userId}_${Date.now()}`;
      subscription.stripeSubscriptionId = `sub_mock_${userId}_${Date.now()}`;
    }

    this.storage.subscriptions.push(subscription);

    // Create billing event
    await this.createBillingEvent(userId, 'subscription_created', {
      planId,
      subscriptionId: subscription.stripeSubscriptionId,
      amount: plan.price
    });

    // Initialize usage tracking
    await this.initializeUsageTracking(userId, planId);

    await this.saveData();
    return subscription;
  }

  // Get user's current subscription
  async getUserSubscription(userId: string): Promise<UserSubscription | null> {
    const subscription = this.storage.subscriptions.find(
      sub => sub.userId === userId && sub.status === 'active'
    );
    return subscription || null;
  }

  // Update subscription
  async updateSubscription(userId: string, planId: string): Promise<UserSubscription> {
    const currentSub = await this.getUserSubscription(userId);
    if (!currentSub) {
      throw new NotFoundException('No active subscription found');
    }

    const newPlan = await this.getBillingPlan(planId);

    // Update subscription
    currentSub.planId = planId;
    currentSub.updatedAt = new Date();

    // Create billing event
    await this.createBillingEvent(userId, 'subscription_updated', {
      oldPlanId: currentSub.planId,
      newPlanId: planId,
      amount: newPlan.price
    });

    await this.saveData();
    return currentSub;
  }

  // Cancel subscription
  async cancelSubscription(userId: string, cancelImmediately: boolean = false): Promise<UserSubscription> {
    const subscription = await this.getUserSubscription(userId);
    if (!subscription) {
      throw new NotFoundException('No active subscription found');
    }

    if (cancelImmediately) {
      subscription.status = 'canceled';
    } else {
      subscription.cancelAtPeriodEnd = true;
    }
    subscription.updatedAt = new Date();

    // Create billing event
    await this.createBillingEvent(userId, 'subscription_canceled', {
      cancelImmediately,
      canceledAt: new Date()
    });

    await this.saveData();
    return subscription;
  }

  // Check if user can make request (rate limiting)
  async canMakeRequest(userId: string): Promise<{ allowed: boolean; resetAt?: Date; remaining?: number }> {
    const subscription = await this.getUserSubscription(userId);
    const plan = subscription ? await this.getBillingPlan(subscription.planId) : await this.getBillingPlan('free');

    // Get current usage
    const usage = this.storage.usage.find(u => u.userId === userId);
    
    if (!usage) {
      // Initialize usage if not exists
      await this.initializeUsageTracking(userId, plan.id);
      return { allowed: true, remaining: plan.limits.requestsPerHour - 1 };
    }

    // Check if usage period has reset
    if (new Date() > usage.resetAt) {
      usage.requests = 0;
      usage.resetAt = new Date(Date.now() + 60 * 60 * 1000); // Reset in 1 hour
      usage.timestamp = new Date();
    }

    // Check if under limit
    if (usage.requests < plan.limits.requestsPerHour) {
      usage.requests++;
      usage.timestamp = new Date();
      await this.saveData();
      
      return {
        allowed: true,
        remaining: plan.limits.requestsPerHour - usage.requests,
        resetAt: usage.resetAt
      };
    }

    return {
      allowed: false,
      resetAt: usage.resetAt,
      remaining: 0
    };
  }

  // Initialize usage tracking for user
  private async initializeUsageTracking(userId: string, planId: string): Promise<void> {
    const existingUsage = this.storage.usage.find(u => u.userId === userId);
    if (existingUsage) {
      existingUsage.planId = planId;
      existingUsage.requests = 0;
      existingUsage.resetAt = new Date(Date.now() + 60 * 60 * 1000);
      existingUsage.timestamp = new Date();
    } else {
      const usage: UsageRecord = {
        userId,
        planId,
        requests: 0,
        storageUsedGB: 0,
        timestamp: new Date(),
        resetAt: new Date(Date.now() + 60 * 60 * 1000)
      };
      this.storage.usage.push(usage);
    }
  }

  // Create billing event
  private async createBillingEvent(userId: string, type: BillingEvent['type'], data: any): Promise<void> {
    const event: BillingEvent = {
      id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId,
      type,
      data,
      timestamp: new Date()
    };
    this.storage.events.push(event);
  }

  // Get usage statistics
  async getUsageStats(userId: string): Promise<{
    currentUsage: UsageRecord;
    plan: BillingPlan;
    subscription: UserSubscription | null;
  }> {
    const subscription = await this.getUserSubscription(userId);
    const plan = subscription ? await this.getBillingPlan(subscription.planId) : await this.getBillingPlan('free');
    const usage = this.storage.usage.find(u => u.userId === userId);

    if (!usage) {
      await this.initializeUsageTracking(userId, plan.id);
      const newUsage = this.storage.usage.find(u => u.userId === userId)!;
      return { currentUsage: newUsage, plan, subscription };
    }

    return { currentUsage: usage, plan, subscription };
  }

  // Get billing events for user
  async getBillingEvents(userId: string, limit: number = 50): Promise<BillingEvent[]> {
    return this.storage.events
      .filter(event => event.userId === userId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  // Webhook handler for Stripe events (mock implementation)
  async handleStripeWebhook(eventType: string, eventData: any): Promise<void> {
    console.log(`Processing Stripe webhook: ${eventType}`, eventData);

    switch (eventType) {
      case 'invoice.payment_succeeded':
        await this.handlePaymentSucceeded(eventData);
        break;
      case 'invoice.payment_failed':
        await this.handlePaymentFailed(eventData);
        break;
      case 'customer.subscription.deleted':
        await this.handleSubscriptionDeleted(eventData);
        break;
      default:
        console.log(`Unhandled webhook event: ${eventType}`);
    }
  }

  private async handlePaymentSucceeded(eventData: any): Promise<void> {
    const customerId = eventData.customer;
    const subscription = this.storage.subscriptions.find(
      sub => sub.stripeCustomerId === customerId
    );

    if (subscription) {
      subscription.status = 'active';
      subscription.updatedAt = new Date();

      await this.createBillingEvent(subscription.userId, 'payment_succeeded', {
        amount: eventData.amount_paid,
        currency: eventData.currency
      });

      await this.saveData();
    }
  }

  private async handlePaymentFailed(eventData: any): Promise<void> {
    const customerId = eventData.customer;
    const subscription = this.storage.subscriptions.find(
      sub => sub.stripeCustomerId === customerId
    );

    if (subscription) {
      subscription.status = 'past_due';
      subscription.updatedAt = new Date();

      await this.createBillingEvent(subscription.userId, 'payment_failed', {
        amount: eventData.amount_due,
        currency: eventData.currency,
        failure_reason: eventData.failure_reason
      });

      await this.saveData();
    }
  }

  private async handleSubscriptionDeleted(eventData: any): Promise<void> {
    const customerId = eventData.customer;
    const subscription = this.storage.subscriptions.find(
      sub => sub.stripeCustomerId === customerId
    );

    if (subscription) {
      subscription.status = 'canceled';
      subscription.updatedAt = new Date();

      await this.createBillingEvent(subscription.userId, 'subscription_canceled', {
        canceled_at: new Date(),
        reason: 'subscription_deleted'
      });

      await this.saveData();
    }
  }
}