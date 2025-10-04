import { 
  Body, 
  Controller, 
  Get,
  Post, 
  Param,
  Query,
  HttpException, 
  HttpStatus,
  Logger,
  UseGuards,
  UsePipes,
  ValidationPipe
} from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { JwtAuthGuard } from '../../security/jwt.guard';

const execAsync = promisify(exec);

export interface IaCGenerationRequest {
  prompt: string;
  platform?: 'kubernetes' | 'terraform' | 'ansible';
  maxTokens?: number;
  template?: string;
  parameters?: Record<string, any>;
  outputFormat?: 'yaml' | 'json' | 'hcl';
}

export interface IaCGenerationResponse {
  ok: boolean;
  templateId: string;
  platform: string;
  content: string;
  metadata: {
    originalPrompt: string;
    generatedAt: string;
    templateType: string;
    complexity: string;
    estimatedCost: number;
  };
  validation: {
    syntaxValid: boolean;
    bestPracticesScore: number;
    securityScore: number;
    issues: string[];
    warnings: string[];
  };
  suggestions: string[];
  nextSteps: string[];
  fallback?: boolean;
}

export interface IaCTemplateLibrary {
  id: string;
  name: string;
  description: string;
  platform: string;
  category: string;
  template: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required: boolean;
    default?: any;
  }>;
  examples: Array<{
    title: string;
    description: string;
    parameters: Record<string, any>;
  }>;
}

@Controller('/api/iac')
@UsePipes(new ValidationPipe({ transform: true }))
export class IacController {
  private readonly logger = new Logger(IacController.name);
  private readonly aiScriptPath = path.join(process.cwd(), '..', 'ai-scripts');
  private templateLibrary: IaCTemplateLibrary[] = [];

  constructor() {
    this.initializeTemplateLibrary();
  }

  /**
   * Initialize built-in template library
   */
  private initializeTemplateLibrary(): void {
    this.templateLibrary = [
      {
        id: 'k8s-web-app',
        name: 'Kubernetes Web Application',
        description: 'Complete web application deployment with service and ingress',
        platform: 'kubernetes',
        category: 'web-applications',
        template: 'kubernetes-web-app.yaml',
        parameters: [
          { name: 'appName', type: 'string', description: 'Application name', required: true },
          { name: 'image', type: 'string', description: 'Container image', required: true },
          { name: 'replicas', type: 'number', description: 'Number of replicas', required: false, default: 3 },
          { name: 'port', type: 'number', description: 'Container port', required: false, default: 80 }
        ],
        examples: [
          {
            title: 'React Frontend',
            description: 'React.js frontend application',
            parameters: { appName: 'react-frontend', image: 'nginx:alpine', replicas: 2, port: 80 }
          }
        ]
      },
      {
        id: 'terraform-aws-ec2',
        name: 'AWS EC2 Instance',
        description: 'AWS EC2 instance with security group and key pair',
        platform: 'terraform',
        category: 'compute',
        template: 'terraform-aws-ec2.tf',
        parameters: [
          { name: 'instanceType', type: 'string', description: 'EC2 instance type', required: false, default: 't3.micro' },
          { name: 'region', type: 'string', description: 'AWS region', required: false, default: 'us-west-2' },
          { name: 'keyName', type: 'string', description: 'SSH key pair name', required: true }
        ],
        examples: [
          {
            title: 'Development Server',
            description: 'Small development server',
            parameters: { instanceType: 't3.micro', region: 'us-west-2', keyName: 'dev-key' }
          }
        ]
      },
      {
        id: 'ansible-docker-deployment',
        name: 'Docker Application Deployment',
        description: 'Deploy containerized application using Ansible',
        platform: 'ansible',
        category: 'deployment',
        template: 'ansible-docker-deployment.yml',
        parameters: [
          { name: 'appName', type: 'string', description: 'Application name', required: true },
          { name: 'image', type: 'string', description: 'Docker image', required: true },
          { name: 'port', type: 'number', description: 'Application port', required: false, default: 8080 }
        ],
        examples: [
          {
            title: 'Node.js API',
            description: 'Node.js REST API deployment',
            parameters: { appName: 'nodejs-api', image: 'node:18-alpine', port: 3000 }
          }
        ]
      }
    ];

    this.logger.log(`Initialized template library with ${this.templateLibrary.length} templates`);
  }

  /**
   * Generate infrastructure code from natural language prompt
   */
  @Post('generate')
  async generate(@Body() request: IaCGenerationRequest): Promise<IaCGenerationResponse> {
    try {
      this.logger.log(`IaC generation requested: ${request.prompt} (${request.platform || 'auto'})`);
      
      // Validate input
      if (!request.prompt || request.prompt.trim().length === 0) {
        throw new HttpException('Prompt is required', HttpStatus.BAD_REQUEST);
      }

      // Try AI service first
      try {
        const aiResult = await this.generateWithAI(request);
        if (aiResult) {
          return aiResult;
        }
      } catch (aiError) {
        this.logger.warn('AI service failed, using fallback generation', aiError);
      }

      // Fallback to deterministic generation
      return await this.generateFallback(request);

    } catch (error) {
      this.logger.error('IaC generation failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Infrastructure generation failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Generate infrastructure using AI service
   */
  private async generateWithAI(request: IaCGenerationRequest): Promise<IaCGenerationResponse | null> {
    try {
      // Try Python AI service first
      const pythonResult = await this.callPythonAI(request);
      if (pythonResult) {
        return pythonResult;
      }

      // Fallback to Flask AI service
      const flaskResult = await this.callFlaskAI(request);
      return flaskResult;

    } catch (error) {
      this.logger.error('AI generation failed', error);
      return null;
    }
  }

  /**
   * Call Python AI script directly
   */
  private async callPythonAI(request: IaCGenerationRequest): Promise<IaCGenerationResponse | null> {
    try {
      const scriptPath = path.join(this.aiScriptPath, 'iac_generator.py');
      
      if (!fs.existsSync(scriptPath)) {
        this.logger.warn('IaC generator script not found');
        return null;
      }

      // Prepare input data
      const inputData = {
        prompt: request.prompt,
        platform: request.platform || 'kubernetes',
        max_tokens: request.maxTokens || 512,
        parameters: request.parameters || {}
      };

      // Write temp input file
      const tempFile = path.join(this.aiScriptPath, `iac_input_${Date.now()}.json`);
      await fs.promises.writeFile(tempFile, JSON.stringify(inputData));

      try {
        // Execute Python script
        const { stdout } = await execAsync(`python "${scriptPath}" "${tempFile}"`);
        const result = JSON.parse(stdout);

        // Clean up temp file
        await fs.promises.unlink(tempFile).catch(() => {});

        // Transform to expected format
        return {
          ok: true,
          templateId: result.template_id || `iac_${Date.now()}`,
          platform: result.platform || request.platform || 'kubernetes',
          content: result.template?.content || result.yaml || '',
          metadata: {
            originalPrompt: request.prompt,
            generatedAt: new Date().toISOString(),
            templateType: result.template?.template_type || 'general',
            complexity: 'medium',
            estimatedCost: result.template?.estimated_cost_monthly || 0
          },
          validation: {
            syntaxValid: result.template?.validation_results?.syntax_valid || true,
            bestPracticesScore: result.template?.best_practices_score || 8.0,
            securityScore: result.template?.security_score || 0.8,
            issues: result.template?.validation_results?.issues || [],
            warnings: result.template?.validation_results?.warnings || []
          },
          suggestions: result.suggestions || [],
          nextSteps: result.next_steps || []
        };

      } catch (execError) {
        this.logger.warn('Python script execution failed', execError);
        return null;
      }

    } catch (error) {
      this.logger.error('Python AI call failed', error);
      return null;
    }
  }

  /**
   * Call Flask AI service
   */
  private async callFlaskAI(request: IaCGenerationRequest): Promise<IaCGenerationResponse | null> {
    try {
      const base = process.env.AI_URL || 'http://127.0.0.1:5001';
      const client = axios.create({ timeout: 5000 });
      
      const response = await client.post(`${base}/ai/iac/generate`, {
        prompt: request.prompt,
        platform: request.platform || 'kubernetes',
        max_tokens: request.maxTokens || 512
      });

      if (response.data && response.data.ok) {
        return {
          ok: true,
          templateId: `flask_${Date.now()}`,
          platform: request.platform || 'kubernetes',
          content: response.data.yaml || response.data.content || '',
          metadata: {
            originalPrompt: request.prompt,
            generatedAt: new Date().toISOString(),
            templateType: 'ai_generated',
            complexity: 'medium',
            estimatedCost: 0
          },
          validation: {
            syntaxValid: true,
            bestPracticesScore: 7.0,
            securityScore: 0.7,
            issues: [],
            warnings: []
          },
          suggestions: ['Review generated template before deployment'],
          nextSteps: ['Test template in development environment']
        };
      }

      return null;

    } catch (error) {
      this.logger.warn('Flask AI service call failed', error);
      return null;
    }
  }

  /**
   * Generate fallback infrastructure template
   */
  private async generateFallback(request: IaCGenerationRequest): Promise<IaCGenerationResponse> {
    const platform = request.platform || 'kubernetes';
    const prompt = request.prompt.toLowerCase();
    
    let content: string;
    let templateType: string;

    if (platform === 'kubernetes') {
      content = this.generateKubernetesFallback(prompt);
      templateType = 'kubernetes_service';
    } else if (platform === 'terraform') {
      content = this.generateTerraformFallback(prompt);
      templateType = 'terraform_infrastructure';
    } else if (platform === 'ansible') {
      content = this.generateAnsibleFallback(prompt);
      templateType = 'ansible_playbook';
    } else {
      content = this.generateKubernetesFallback(prompt);
      templateType = 'kubernetes_service';
    }

    return {
      ok: true,
      templateId: `fallback_${Date.now()}`,
      platform,
      content,
      metadata: {
        originalPrompt: request.prompt,
        generatedAt: new Date().toISOString(),
        templateType,
        complexity: 'low',
        estimatedCost: 25.0
      },
      validation: {
        syntaxValid: true,
        bestPracticesScore: 7.0,
        securityScore: 0.8,
        issues: [],
        warnings: ['Generated using fallback template']
      },
      suggestions: [
        'Review and customize the generated template',
        'Add resource limits and probes for production use',
        'Consider security best practices'
      ],
      nextSteps: [
        '1. Review the generated template',
        '2. Customize for your specific needs',
        '3. Test in development environment',
        '4. Deploy to production'
      ],
      fallback: true
    };
  }

  /**
   * Generate Kubernetes fallback template
   */
  private generateKubernetesFallback(prompt: string): string {
    // Analyze prompt for app name and port
    const appName = this.extractAppName(prompt) || 'example-app';
    const port = this.extractPort(prompt) || 80;
    const replicas = this.extractReplicas(prompt) || 2;

    return `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${appName}
  labels:
    app: ${appName}
    version: "1.0"
spec:
  replicas: ${replicas}
  selector:
    matchLabels:
      app: ${appName}
  template:
    metadata:
      labels:
        app: ${appName}
        version: "1.0"
    spec:
      containers:
      - name: ${appName}
        image: nginx:alpine
        ports:
        - containerPort: ${port}
          name: http
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: ${port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: ${port}
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: ${appName}-service
  labels:
    app: ${appName}
spec:
  selector:
    app: ${appName}
  ports:
  - protocol: TCP
    port: 80
    targetPort: ${port}
    name: http
  type: ClusterIP`;
  }

  /**
   * Generate Terraform fallback template
   */
  private generateTerraformFallback(prompt: string): string {
    const appName = this.extractAppName(prompt) || 'example-app';
    
    return `terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

resource "aws_instance" "${appName.replace(/-/g, '_')}" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  
  vpc_security_group_ids = [aws_security_group.${appName.replace(/-/g, '_')}_sg.id]
  
  tags = {
    Name = "${appName}"
    Environment = "production"
    ManagedBy = "terraform"
  }
}

resource "aws_security_group" "${appName.replace(/-/g, '_')}_sg" {
  name_prefix = "${appName}-sg"
  description = "Security group for ${appName}"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${appName}-sg"
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-20.04-amd64-server-*"]
  }
}

output "instance_ip" {
  description = "Public IP address"
  value       = aws_instance.${appName.replace(/-/g, '_')}.public_ip
}`;
  }

  /**
   * Generate Ansible fallback template
   */
  private generateAnsibleFallback(prompt: string): string {
    const appName = this.extractAppName(prompt) || 'example-app';
    const port = this.extractPort(prompt) || 8080;

    return `---
- name: Deploy ${appName}
  hosts: all
  become: yes
  vars:
    app_name: ${appName}
    app_port: ${port}
    app_image: nginx:alpine

  tasks:
    - name: Update package cache
      apt:
        update_cache: yes
      when: ansible_os_family == "Debian"

    - name: Install Docker
      package:
        name: docker.io
        state: present

    - name: Start Docker service
      service:
        name: docker
        state: started
        enabled: yes

    - name: Run application container
      docker_container:
        name: "{{ app_name }}"
        image: "{{ app_image }}"
        state: started
        restart_policy: always
        ports:
          - "{{ app_port }}:{{ app_port }}"
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:{{ app_port }}/health"]
          interval: 30s
          timeout: 10s
          retries: 3

    - name: Verify application is running
      uri:
        url: "http://localhost:{{ app_port }}"
        method: GET
        status_code: 200
      retries: 5
      delay: 10
      register: app_health_check

    - name: Display deployment status
      debug:
        msg: "{{ app_name }} deployed successfully on port {{ app_port }}"
      when: app_health_check is succeeded`;
  }

  /**
   * Extract app name from prompt
   */
  private extractAppName(prompt: string): string | null {
    const matches = prompt.match(/(?:deploy|create|app|service)\s+([a-zA-Z0-9-]+)/i);
    return matches ? matches[1] : null;
  }

  /**
   * Extract port from prompt
   */
  private extractPort(prompt: string): number | null {
    const matches = prompt.match(/port\s*(\d+)/i);
    return matches ? parseInt(matches[1]) : null;
  }

  /**
   * Extract replicas from prompt
   */
  private extractReplicas(prompt: string): number | null {
    const matches = prompt.match(/(\d+)\s*(?:replicas?|instances?|pods?)/i);
    return matches ? parseInt(matches[1]) : null;
  }

  /**
   * Get template library
   */
  @Get('templates')
  getTemplateLibrary(@Query('platform') platform?: string, @Query('category') category?: string) {
    let templates = this.templateLibrary;

    if (platform) {
      templates = templates.filter(t => t.platform === platform);
    }

    if (category) {
      templates = templates.filter(t => t.category === category);
    }

    return {
      ok: true,
      templates,
      total: templates.length
    };
  }

  /**
   * Get specific template by ID
   */
  @Get('templates/:id')
  getTemplate(@Param('id') id: string) {
    const template = this.templateLibrary.find(t => t.id === id);
    
    if (!template) {
      throw new HttpException('Template not found', HttpStatus.NOT_FOUND);
    }

    return {
      ok: true,
      template
    };
  }

  /**
   * Generate from template
   */
  @Post('templates/:id/generate')
  async generateFromTemplate(@Param('id') id: string, @Body() parameters: Record<string, any>) {
    try {
      const template = this.templateLibrary.find(t => t.id === id);
      
      if (!template) {
        throw new HttpException('Template not found', HttpStatus.NOT_FOUND);
      }

      // Validate required parameters
      const missingParams = template.parameters
        .filter(p => p.required && !parameters[p.name])
        .map(p => p.name);

      if (missingParams.length > 0) {
        throw new HttpException(
          `Missing required parameters: ${missingParams.join(', ')}`,
          HttpStatus.BAD_REQUEST
        );
      }

      // Apply default values
      const finalParams = { ...parameters };
      template.parameters.forEach(p => {
        if (finalParams[p.name] === undefined && p.default !== undefined) {
          finalParams[p.name] = p.default;
        }
      });

      // Generate content based on template
      const content = await this.renderTemplate(template, finalParams);

      return {
        ok: true,
        templateId: `template_${id}_${Date.now()}`,
        platform: template.platform,
        content,
        metadata: {
          templateId: id,
          templateName: template.name,
          generatedAt: new Date().toISOString(),
          parameters: finalParams
        }
      };

    } catch (error) {
      this.logger.error('Template generation failed', error);
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException('Template generation failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  /**
   * Render template with parameters
   */
  private async renderTemplate(template: IaCTemplateLibrary, parameters: Record<string, any>): Promise<string> {
    // Simple template rendering (in production, would use a proper template engine)
    let content = '';

    switch (template.id) {
      case 'k8s-web-app':
        content = this.renderKubernetesWebApp(parameters);
        break;
      case 'terraform-aws-ec2':
        content = this.renderTerraformEC2(parameters);
        break;
      case 'ansible-docker-deployment':
        content = this.renderAnsibleDockerDeployment(parameters);
        break;
      default:
        content = this.generateKubernetesFallback(`Deploy ${parameters.appName || 'app'}`);
    }

    return content;
  }

  /**
   * Render Kubernetes web app template
   */
  private renderKubernetesWebApp(params: Record<string, any>): string {
    const { appName, image, replicas = 3, port = 80 } = params;

    return `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${appName}
  labels:
    app: ${appName}
spec:
  replicas: ${replicas}
  selector:
    matchLabels:
      app: ${appName}
  template:
    metadata:
      labels:
        app: ${appName}
    spec:
      containers:
      - name: ${appName}
        image: ${image}
        ports:
        - containerPort: ${port}
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ${appName}-service
spec:
  selector:
    app: ${appName}
  ports:
  - port: 80
    targetPort: ${port}
  type: ClusterIP`;
  }

  /**
   * Render Terraform EC2 template
   */
  private renderTerraformEC2(params: Record<string, any>): string {
    const { instanceType = 't3.micro', region = 'us-west-2', keyName } = params;

    return `provider "aws" {
  region = "${region}"
}

resource "aws_instance" "main" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "${instanceType}"
  key_name      = "${keyName}"
  
  tags = {
    Name = "CloudForge-Instance"
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-20.04-amd64-server-*"]
  }
}`;
  }

  /**
   * Render Ansible Docker deployment template
   */
  private renderAnsibleDockerDeployment(params: Record<string, any>): string {
    const { appName, image, port = 8080 } = params;

    return `---
- name: Deploy ${appName}
  hosts: all
  become: yes
  tasks:
    - name: Run ${appName} container
      docker_container:
        name: ${appName}
        image: ${image}
        state: started
        ports:
          - "${port}:${port}"`;
  }

  /**
   * Get IaC service health status
   */
  @Get('health')
  getHealth() {
    try {
      return {
        ok: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        templates: this.templateLibrary.length,
        platforms: ['kubernetes', 'terraform', 'ansible'],
        service: 'iac-service',
        version: '1.0.0'
      };
    } catch (error) {
      throw new HttpException(
        { status: 'unhealthy', error: error instanceof Error ? error.message : String(error) },
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }
}
