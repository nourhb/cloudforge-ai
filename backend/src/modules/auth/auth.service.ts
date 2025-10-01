import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import { JwtServiceLite } from '../../security/jwt.service';
import * as crypto from 'crypto';

/**
 * Simple password hashing utility using Node.js crypto
 */
class PasswordUtil {
  private static readonly saltLength = 32;
  private static readonly iterations = 100000;
  private static readonly keyLength = 64;

  static async hash(password: string): Promise<string> {
    const salt = crypto.randomBytes(this.saltLength);
    const hash = crypto.pbkdf2Sync(password, salt, this.iterations, this.keyLength, 'sha256');
    return `${salt.toString('hex')}:${hash.toString('hex')}`;
  }

  static async compare(password: string, hashedPassword: string): Promise<boolean> {
    const [saltHex, hashHex] = hashedPassword.split(':');
    if (!saltHex || !hashHex) return false;
    
    const salt = Buffer.from(saltHex, 'hex');
    const hash = crypto.pbkdf2Sync(password, salt, this.iterations, this.keyLength, 'sha256');
    const expectedHash = Buffer.from(hashHex, 'hex');
    
    return crypto.timingSafeEqual(hash, expectedHash);
  }
}

interface OtpEntry {
  code: string;
  expiresAt: number;
  attempts: number;
  maxAttempts: number;
}

interface User {
  id: string;
  email: string;
  username: string;
  passwordHash?: string;
  role: UserRole;
  permissions: string[];
  profile: UserProfile;
  security: UserSecurity;
  createdAt: string;
  lastLoginAt?: string;
  isActive: boolean;
  emailVerified: boolean;
  twoFactorEnabled: boolean;
}

interface UserProfile {
  firstName: string;
  lastName: string;
  company?: string;
  position?: string;
  avatar?: string;
  timezone: string;
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    language: string;
    notifications: {
      email: boolean;
      push: boolean;
      slack: boolean;
    };
  };
}

interface UserSecurity {
  lastPasswordChange: string;
  failedLoginAttempts: number;
  lockedUntil?: string;
  ipWhitelist: string[];
  sessionTimeoutMinutes: number;
  requireTwoFactor: boolean;
}

enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  VIEWER = 'viewer',
  DEVELOPER = 'developer',
  OPERATOR = 'operator'
}

interface LoginRequest {
  email: string;
  password?: string;
  otpCode?: string;
  rememberMe?: boolean;
}

interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  firstName: string;
  lastName: string;
  company?: string;
  acceptTerms: boolean;
}

interface AuthResponse {
  success: boolean;
  user?: Partial<User>;
  token?: string;
  refreshToken?: string;
  expiresIn?: number;
  requiresOtp?: boolean;
  message?: string;
}

interface SessionInfo {
  userId: string;
  email: string;
  role: UserRole;
  permissions: string[];
  sessionId: string;
  createdAt: string;
  expiresAt: string;
  ipAddress: string;
  userAgent: string;
}

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);
  private otpStore = new Map<string, OtpEntry>();
  private userStore = new Map<string, User>();
  private sessionStore = new Map<string, SessionInfo>();
  private readonly otpTtlMs = 5 * 60 * 1000; // 5 minutes
  private readonly maxLoginAttempts = 5;
  private readonly lockoutDurationMs = 30 * 60 * 1000; // 30 minutes

  constructor(private readonly jwtService: JwtServiceLite) {
    this.initializeDefaultUsers();
  }

  /**
   * Initialize default admin user
   */
  private async initializeDefaultUsers(): Promise<void> {
    const adminUser: User = {
      id: crypto.randomUUID(),
      email: 'admin@cloudforge.ai',
      username: 'admin',
      passwordHash: await PasswordUtil.hash('CloudForge2025!'),
      role: UserRole.ADMIN,
      permissions: [
        'users:read', 'users:write', 'users:delete',
        'migration:read', 'migration:write', 'migration:execute',
        'marketplace:read', 'marketplace:write', 'marketplace:execute',
        'iac:read', 'iac:write', 'iac:execute',
        'system:admin', 'system:monitoring'
      ],
      profile: {
        firstName: 'System',
        lastName: 'Administrator',
        company: 'CloudForge AI',
        position: 'System Administrator',
        timezone: 'UTC',
        preferences: {
          theme: 'dark',
          language: 'en',
          notifications: {
            email: true,
            push: true,
            slack: false
          }
        }
      },
      security: {
        lastPasswordChange: new Date().toISOString(),
        failedLoginAttempts: 0,
        ipWhitelist: [],
        sessionTimeoutMinutes: 480, // 8 hours
        requireTwoFactor: false
      },
      createdAt: new Date().toISOString(),
      isActive: true,
      emailVerified: true,
      twoFactorEnabled: false
    };

    this.userStore.set(adminUser.email, adminUser);
    this.logger.log('Default admin user initialized');
  }

  /**
   * Generate OTP for authentication
   */
  generateOtp(identifier: string): string {
    const code = String(Math.floor(100000 + Math.random() * 900000));
    const expiresAt = Date.now() + this.otpTtlMs;
    
    this.otpStore.set(identifier, { 
      code, 
      expiresAt,
      attempts: 0,
      maxAttempts: 3
    });
    
    this.logger.log(`OTP generated for ${identifier}`);
    return code;
  }

  /**
   * Verify OTP code
   */
  verifyOtp(identifier: string, code: string): boolean {
    const entry = this.otpStore.get(identifier);
    if (!entry) return false;
    
    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.otpStore.delete(identifier);
      return false;
    }
    
    // Check attempts
    entry.attempts++;
    if (entry.attempts > entry.maxAttempts) {
      this.otpStore.delete(identifier);
      this.logger.warn(`OTP max attempts exceeded for ${identifier}`);
      return false;
    }
    
    // Verify code
    const isValid = entry.code === code;
    if (isValid) {
      this.otpStore.delete(identifier);
    }
    
    return isValid;
  }

  /**
   * Register new user
   */
  async register(request: RegisterRequest): Promise<AuthResponse> {
    try {
      // Validate input
      if (!request.acceptTerms) {
        throw new HttpException('Terms and conditions must be accepted', HttpStatus.BAD_REQUEST);
      }

      // Check if user already exists
      if (this.userStore.has(request.email)) {
        throw new HttpException('User already exists', HttpStatus.CONFLICT);
      }

      // Validate password strength
      if (!this.isPasswordStrong(request.password)) {
        throw new HttpException(
          'Password must be at least 8 characters with uppercase, lowercase, number and special character',
          HttpStatus.BAD_REQUEST
        );
      }

      // Hash password
      const passwordHash = await PasswordUtil.hash(request.password);

      // Create user
      const user: User = {
        id: crypto.randomUUID(),
        email: request.email,
        username: request.username,
        passwordHash,
        role: UserRole.USER,
        permissions: [
          'migration:read', 'migration:write',
          'marketplace:read', 'marketplace:execute',
          'iac:read', 'iac:write'
        ],
        profile: {
          firstName: request.firstName,
          lastName: request.lastName,
          company: request.company,
          timezone: 'UTC',
          preferences: {
            theme: 'light',
            language: 'en',
            notifications: {
              email: true,
              push: false,
              slack: false
            }
          }
        },
        security: {
          lastPasswordChange: new Date().toISOString(),
          failedLoginAttempts: 0,
          ipWhitelist: [],
          sessionTimeoutMinutes: 240, // 4 hours
          requireTwoFactor: false
        },
        createdAt: new Date().toISOString(),
        isActive: true,
        emailVerified: false,
        twoFactorEnabled: false
      };

      this.userStore.set(user.email, user);

      // Generate email verification OTP
      const verificationCode = this.generateOtp(user.email);
      
      this.logger.log(`User registered: ${user.email}`);

      return {
        success: true,
        user: this.sanitizeUser(user),
        message: `Registration successful. Verification code sent to ${user.email}: ${verificationCode}`
      };

    } catch (error) {
      this.logger.error('Registration failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Registration failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Authenticate user login
   */
  async login(request: LoginRequest, clientInfo: { ip: string; userAgent: string }): Promise<AuthResponse> {
    try {
      const user = this.userStore.get(request.email);
      
      if (!user) {
        throw new HttpException('Invalid credentials', HttpStatus.UNAUTHORIZED);
      }

      // Check if account is locked
      if (this.isAccountLocked(user)) {
        throw new HttpException('Account temporarily locked due to failed login attempts', HttpStatus.LOCKED);
      }

      // Check if account is active
      if (!user.isActive) {
        throw new HttpException('Account is disabled', HttpStatus.FORBIDDEN);
      }

      // Verify password
      if (user.passwordHash && request.password) {
        const isPasswordValid = await PasswordUtil.compare(request.password, user.passwordHash);
        
        if (!isPasswordValid) {
          await this.handleFailedLogin(user);
          throw new HttpException('Invalid credentials', HttpStatus.UNAUTHORIZED);
        }
      }

      // Check if OTP is required
      if (user.twoFactorEnabled || user.security.requireTwoFactor) {
        if (!request.otpCode) {
          // Generate and send OTP
          const otpCode = this.generateOtp(user.email);
          this.logger.log(`OTP required for ${user.email}: ${otpCode}`);
          
          return {
            success: false,
            requiresOtp: true,
            message: `OTP required. Code sent to ${user.email}: ${otpCode}`
          };
        }

        // Verify OTP
        if (!this.verifyOtp(user.email, request.otpCode)) {
          throw new HttpException('Invalid or expired OTP', HttpStatus.UNAUTHORIZED);
        }
      }

      // Reset failed login attempts
      user.security.failedLoginAttempts = 0;
      user.security.lockedUntil = undefined;
      user.lastLoginAt = new Date().toISOString();

      // Generate tokens
      const sessionId = crypto.randomUUID();
      const tokenPayload = {
        userId: user.id,
        email: user.email,
        role: user.role,
        sessionId
      };

      const token = this.jwtService.generateToken(tokenPayload);
      const refreshToken = this.jwtService.generateRefreshToken(tokenPayload);

      // Create session
      const expiresIn = request.rememberMe ? 30 * 24 * 60 * 60 : user.security.sessionTimeoutMinutes * 60; // 30 days or user preference
      const sessionInfo: SessionInfo = {
        userId: user.id,
        email: user.email,
        role: user.role,
        permissions: user.permissions,
        sessionId,
        createdAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + expiresIn * 1000).toISOString(),
        ipAddress: clientInfo.ip,
        userAgent: clientInfo.userAgent
      };

      this.sessionStore.set(sessionId, sessionInfo);

      this.logger.log(`User logged in: ${user.email}`);

      return {
        success: true,
        user: this.sanitizeUser(user),
        token,
        refreshToken,
        expiresIn
      };

    } catch (error) {
      this.logger.error('Login failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Login failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Logout user and invalidate session
   */
  async logout(sessionId: string): Promise<{ success: boolean; message: string }> {
    try {
      const session = this.sessionStore.get(sessionId);
      
      if (session) {
        this.sessionStore.delete(sessionId);
        this.logger.log(`User logged out: ${session.email}`);
      }

      return {
        success: true,
        message: 'Logged out successfully'
      };

    } catch (error) {
      this.logger.error('Logout failed', error);
      throw new HttpException('Logout failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Refresh access token
   */
  async refreshToken(refreshToken: string): Promise<{ token: string; expiresIn: number }> {
    try {
      const payload = this.jwtService.verifyRefreshToken(refreshToken);
      const session = this.sessionStore.get(payload.sessionId);

      if (!session) {
        throw new HttpException('Invalid session', HttpStatus.UNAUTHORIZED);
      }

      // Check session expiration
      if (new Date() > new Date(session.expiresAt)) {
        this.sessionStore.delete(payload.sessionId);
        throw new HttpException('Session expired', HttpStatus.UNAUTHORIZED);
      }

      // Generate new access token
      const newToken = this.jwtService.generateToken({
        userId: session.userId,
        email: session.email,
        role: session.role,
        sessionId: session.sessionId
      });

      return {
        token: newToken,
        expiresIn: 15 * 60 // 15 minutes
      };

    } catch (error) {
      this.logger.error('Token refresh failed', error);
      throw new HttpException('Token refresh failed', HttpStatus.UNAUTHORIZED);
    }
  }

  /**
   * Get user by ID
   */
  getUserById(userId: string): User | null {
    for (const user of this.userStore.values()) {
      if (user.id === userId) {
        return user;
      }
    }
    return null;
  }

  /**
   * Get user by email
   */
  getUserByEmail(email: string): User | null {
    return this.userStore.get(email) || null;
  }

  /**
   * Update user profile
   */
  async updateProfile(userId: string, updates: Partial<UserProfile>): Promise<User> {
    const user = this.getUserById(userId);
    
    if (!user) {
      throw new HttpException('User not found', HttpStatus.NOT_FOUND);
    }

    // Update profile
    user.profile = { ...user.profile, ...updates };
    this.userStore.set(user.email, user);

    this.logger.log(`User profile updated: ${user.email}`);
    return user;
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, currentPassword: string, newPassword: string): Promise<{ success: boolean }> {
    const user = this.getUserById(userId);
    
    if (!user || !user.passwordHash) {
      throw new HttpException('User not found', HttpStatus.NOT_FOUND);
    }

    // Verify current password
    const isCurrentPasswordValid = await PasswordUtil.compare(currentPassword, user.passwordHash);
    if (!isCurrentPasswordValid) {
      throw new HttpException('Current password is incorrect', HttpStatus.BAD_REQUEST);
    }

    // Validate new password strength
    if (!this.isPasswordStrong(newPassword)) {
      throw new HttpException(
        'Password must be at least 8 characters with uppercase, lowercase, number and special character',
        HttpStatus.BAD_REQUEST
      );
    }

    // Hash and update password
    user.passwordHash = await PasswordUtil.hash(newPassword);
    user.security.lastPasswordChange = new Date().toISOString();
    this.userStore.set(user.email, user);

    this.logger.log(`Password changed for user: ${user.email}`);
    return { success: true };
  }

  /**
   * List all users (admin only)
   */
  listUsers(limit: number = 50, offset: number = 0): { users: Partial<User>[]; total: number } {
    const allUsers = Array.from(this.userStore.values());
    const paginated = allUsers.slice(offset, offset + limit);
    
    return {
      users: paginated.map(u => this.sanitizeUser(u)),
      total: allUsers.length
    };
  }

  /**
   * Get active sessions for user
   */
  getUserSessions(userId: string): SessionInfo[] {
    return Array.from(this.sessionStore.values())
      .filter(s => s.userId === userId)
      .filter(s => new Date() <= new Date(s.expiresAt));
  }

  /**
   * Revoke user session
   */
  revokeSession(sessionId: string): { success: boolean } {
    const deleted = this.sessionStore.delete(sessionId);
    return { success: deleted };
  }

  /**
   * Check if password meets strength requirements
   */
  private isPasswordStrong(password: string): boolean {
    const minLength = 8;
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    return password.length >= minLength && hasUppercase && hasLowercase && hasNumber && hasSpecialChar;
  }

  /**
   * Check if account is locked
   */
  private isAccountLocked(user: User): boolean {
    if (!user.security.lockedUntil) {
      return false;
    }
    
    const lockExpired = new Date() > new Date(user.security.lockedUntil);
    if (lockExpired) {
      user.security.lockedUntil = undefined;
      user.security.failedLoginAttempts = 0;
      return false;
    }
    
    return true;
  }

  /**
   * Handle failed login attempt
   */
  private async handleFailedLogin(user: User): Promise<void> {
    user.security.failedLoginAttempts++;
    
    if (user.security.failedLoginAttempts >= this.maxLoginAttempts) {
      user.security.lockedUntil = new Date(Date.now() + this.lockoutDurationMs).toISOString();
      this.logger.warn(`Account locked due to failed login attempts: ${user.email}`);
    }
    
    this.userStore.set(user.email, user);
  }

  /**
   * Sanitize user data for response
   */
  private sanitizeUser(user: User): Partial<User> {
    const { passwordHash, ...sanitized } = user;
    return sanitized;
  }

  /**
   * Get authentication service health
   */
  getHealth(): { status: string; users: number; activeSessions: number } {
    const now = new Date();
    const activeSessions = Array.from(this.sessionStore.values())
      .filter(s => new Date(s.expiresAt) > now).length;

    return {
      status: 'healthy',
      users: this.userStore.size,
      activeSessions
    };
  }

  /**
   * Validate user permissions
   */
  hasPermission(user: User, permission: string): boolean {
    return user.permissions.includes(permission) || user.role === UserRole.ADMIN;
  }

  /**
   * Get user roles and permissions
   */
  getRolePermissions(): Record<UserRole, string[]> {
    return {
      [UserRole.ADMIN]: [
        'users:read', 'users:write', 'users:delete',
        'migration:read', 'migration:write', 'migration:execute',
        'marketplace:read', 'marketplace:write', 'marketplace:execute',
        'iac:read', 'iac:write', 'iac:execute',
        'system:admin', 'system:monitoring'
      ],
      [UserRole.DEVELOPER]: [
        'migration:read', 'migration:write', 'migration:execute',
        'marketplace:read', 'marketplace:write', 'marketplace:execute',
        'iac:read', 'iac:write', 'iac:execute'
      ],
      [UserRole.OPERATOR]: [
        'migration:read', 'migration:execute',
        'marketplace:read', 'marketplace:execute',
        'iac:read', 'iac:execute',
        'system:monitoring'
      ],
      [UserRole.USER]: [
        'migration:read', 'migration:write',
        'marketplace:read', 'marketplace:execute',
        'iac:read', 'iac:write'
      ],
      [UserRole.VIEWER]: [
        'migration:read',
        'marketplace:read',
        'iac:read'
      ]
    };
  }
}
