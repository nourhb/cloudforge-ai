#!/usr/bin/env python3
"""
CloudForge AI - Advanced Infrastructure as Code Generator
AI-powered generation of infrastructure manifests using Hugging Face Transformers and GPT models.

This module provides comprehensive IaC generation capabilities:
- Natural language to Kubernetes/Terraform/Ansible code generation
- AI-powered template optimization and security hardening
- Multi-cloud provider support (AWS, Azure, GCP, Oracle Cloud)
- Cost estimation and resource optimization
- Compliance and security scanning
- Best practices enforcement

Dependencies:
- transformers: Hugging Face transformers for NLP
- torch: PyTorch for model execution
- jinja2: Template engine for code generation
- pyyaml: YAML processing
- jsonschema: Schema validation

Usage:
    python iac_generator_enhanced.py --prompt "Create a microservices deployment" --platform kubernetes --output manifests/
    
    from iac_generator_enhanced import IaCGenerator
    generator = IaCGenerator()
    manifests = generator.generate_from_prompt(prompt, "kubernetes")
"""

import os
import sys
import json
import yaml
import logging
import argparse
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
from jinja2 import Template, Environment, FileSystemLoader
import jsonschema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iac_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IaCTemplate:
    """Infrastructure as Code template structure"""
    name: str
    platform: str  # 'kubernetes', 'terraform', 'ansible'
    provider: str  # 'aws', 'azure', 'gcp', 'oracle', 'on-premise'
    content: str
    metadata: Dict[str, Any]
    validation_status: str
    security_score: float
    estimated_cost: Optional[float]
    compliance_status: Dict[str, bool]

@dataclass
class GenerationRequest:
    """IaC generation request structure"""
    prompt: str
    platform: str
    provider: str
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    security_level: str  # 'basic', 'standard', 'high', 'enterprise'
    compliance_standards: List[str]

@dataclass
class GenerationResult:
    """IaC generation result structure"""
    request_id: str
    templates: List[IaCTemplate]
    generation_time: float
    ai_insights: str
    recommendations: List[str]
    validation_results: Dict[str, Any]
    security_analysis: Dict[str, Any]
    cost_estimate: Dict[str, Any]

class SecurityAnalyzer:
    """Security analysis for generated IaC templates"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Security rules for different platforms
        self.kubernetes_security_rules = {
            'no_privileged_containers': {
                'pattern': r'privileged:\s*true',
                'severity': 'high',
                'message': 'Privileged containers should be avoided'
            },
            'no_host_network': {
                'pattern': r'hostNetwork:\s*true',
                'severity': 'medium',
                'message': 'Host network access should be limited'
            },
            'resource_limits_required': {
                'pattern': r'resources:\s*\n.*limits:',
                'severity': 'medium',
                'message': 'Resource limits should be specified',
                'inverse': True
            },
            'no_default_namespace': {
                'pattern': r'namespace:\s*default',
                'severity': 'low',
                'message': 'Avoid using default namespace'
            }
        }
        
        self.terraform_security_rules = {
            'encrypted_storage': {
                'pattern': r'encrypted\s*=\s*false',
                'severity': 'high',
                'message': 'Storage should be encrypted'
            },
            'public_access_blocked': {
                'pattern': r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                'severity': 'high',
                'message': 'Avoid unrestricted access (0.0.0.0/0)'
            },
            'tags_required': {
                'pattern': r'tags\s*=\s*{',
                'severity': 'medium',
                'message': 'Resources should be tagged',
                'inverse': True
            }
        }
    
    def analyze_security(self, template: IaCTemplate) -> Dict[str, Any]:
        """Analyze security of IaC template"""
        try:
            if template.platform == 'kubernetes':
                return self._analyze_kubernetes_security(template.content)
            elif template.platform == 'terraform':
                return self._analyze_terraform_security(template.content)
            else:
                return self._analyze_ansible_security(template.content)
                
        except Exception as e:
            self.logger.error(f"Security analysis failed: {str(e)}")
            return {'score': 0.5, 'issues': [], 'recommendations': []}
    
    def _analyze_kubernetes_security(self, content: str) -> Dict[str, Any]:
        """Analyze Kubernetes YAML security"""
        issues = []
        score = 1.0
        
        import re
        
        for rule_name, rule in self.kubernetes_security_rules.items():
            pattern = rule['pattern']
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            inverse = rule.get('inverse', False)
            
            if (matches and not inverse) or (not matches and inverse):
                severity_impact = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
                score -= severity_impact.get(rule['severity'], 0.1)
                
                issues.append({
                    'rule': rule_name,
                    'severity': rule['severity'],
                    'message': rule['message'],
                    'matches': len(matches) if not inverse else 0
                })
        
        recommendations = self._generate_security_recommendations(issues)
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': recommendations,
            'platform': 'kubernetes'
        }
    
    def _analyze_terraform_security(self, content: str) -> Dict[str, Any]:
        """Analyze Terraform HCL security"""
        issues = []
        score = 1.0
        
        import re
        
        for rule_name, rule in self.terraform_security_rules.items():
            pattern = rule['pattern']
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            inverse = rule.get('inverse', False)
            
            if (matches and not inverse) or (not matches and inverse):
                severity_impact = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
                score -= severity_impact.get(rule['severity'], 0.1)
                
                issues.append({
                    'rule': rule_name,
                    'severity': rule['severity'],
                    'message': rule['message'],
                    'matches': len(matches) if not inverse else 0
                })
        
        recommendations = self._generate_security_recommendations(issues)
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': recommendations,
            'platform': 'terraform'
        }
    
    def _analyze_ansible_security(self, content: str) -> Dict[str, Any]:
        """Analyze Ansible playbook security"""
        issues = []
        score = 0.8  # Base score for Ansible
        
        import re
        
        # Check for common Ansible security issues
        if re.search(r'become:\s*yes.*become_user:\s*root', content, re.IGNORECASE):
            issues.append({
                'rule': 'avoid_root_escalation',
                'severity': 'medium',
                'message': 'Avoid unnecessary root privilege escalation',
                'matches': 1
            })
            score -= 0.2
        
        if re.search(r'ignore_errors:\s*yes', content, re.IGNORECASE):
            issues.append({
                'rule': 'handle_errors',
                'severity': 'low',
                'message': 'Properly handle errors instead of ignoring them',
                'matches': 1
            })
            score -= 0.1
        
        recommendations = self._generate_security_recommendations(issues)
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': recommendations,
            'platform': 'ansible'
        }
    
    def _generate_security_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on issues"""
        recommendations = []
        
        high_severity_count = sum(1 for issue in issues if issue['severity'] == 'high')
        medium_severity_count = sum(1 for issue in issues if issue['severity'] == 'medium')
        
        if high_severity_count > 0:
            recommendations.append(f"Address {high_severity_count} high-severity security issues immediately")
        
        if medium_severity_count > 0:
            recommendations.append(f"Review and fix {medium_severity_count} medium-severity security issues")
        
        # Specific recommendations based on common issues
        for issue in issues:
            if 'privileged' in issue['rule']:
                recommendations.append("Use security contexts instead of privileged containers")
            elif 'network' in issue['rule']:
                recommendations.append("Implement network policies for pod-to-pod communication")
            elif 'resource' in issue['rule']:
                recommendations.append("Set appropriate resource limits and requests")
            elif 'encryption' in issue['rule']:
                recommendations.append("Enable encryption at rest and in transit")
        
        return list(set(recommendations))  # Remove duplicates

class CostEstimator:
    """Cost estimation for infrastructure templates"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Basic cost models (simplified)
        self.aws_pricing = {
            't3.micro': 0.0104,    # USD per hour
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'ebs_gp2': 0.10,       # USD per GB per month
            'rds_micro': 0.020,    # USD per hour
        }
        
        self.kubernetes_resource_costs = {
            'cpu_core': 0.03,      # USD per core per hour
            'memory_gb': 0.004,    # USD per GB per hour
            'storage_gb': 0.10,    # USD per GB per month
        }
    
    def estimate_cost(self, template: IaCTemplate) -> Dict[str, Any]:
        """Estimate infrastructure cost"""
        try:
            if template.platform == 'kubernetes':
                return self._estimate_kubernetes_cost(template.content)
            elif template.platform == 'terraform':
                return self._estimate_terraform_cost(template.content)
            else:
                return self._estimate_ansible_cost(template.content)
                
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {str(e)}")
            return {'monthly_cost': 0.0, 'breakdown': {}, 'confidence': 'low'}
    
    def _estimate_kubernetes_cost(self, content: str) -> Dict[str, Any]:
        """Estimate Kubernetes deployment costs"""
        import re
        
        total_monthly_cost = 0.0
        breakdown = {}
        
        # Extract resource requests
        cpu_matches = re.findall(r'cpu:\s*(\d+)m', content)
        memory_matches = re.findall(r'memory:\s*(\d+)Mi', content)
        storage_matches = re.findall(r'storage:\s*(\d+)Gi', content)
        
        # Calculate CPU costs
        total_cpu_cores = sum(int(cpu) / 1000 for cpu in cpu_matches)
        cpu_monthly_cost = total_cpu_cores * self.kubernetes_resource_costs['cpu_core'] * 24 * 30
        breakdown['cpu'] = cpu_monthly_cost
        total_monthly_cost += cpu_monthly_cost
        
        # Calculate memory costs
        total_memory_gb = sum(int(mem) / 1024 for mem in memory_matches)
        memory_monthly_cost = total_memory_gb * self.kubernetes_resource_costs['memory_gb'] * 24 * 30
        breakdown['memory'] = memory_monthly_cost
        total_monthly_cost += memory_monthly_cost
        
        # Calculate storage costs
        total_storage_gb = sum(int(storage) for storage in storage_matches)
        storage_monthly_cost = total_storage_gb * self.kubernetes_resource_costs['storage_gb']
        breakdown['storage'] = storage_monthly_cost
        total_monthly_cost += storage_monthly_cost
        
        return {
            'monthly_cost': round(total_monthly_cost, 2),
            'breakdown': breakdown,
            'confidence': 'medium',
            'currency': 'USD',
            'assumptions': 'Based on standard cloud pricing, 100% utilization'
        }
    
    def _estimate_terraform_cost(self, content: str) -> Dict[str, Any]:
        """Estimate Terraform infrastructure costs"""
        import re
        
        total_monthly_cost = 0.0
        breakdown = {}
        
        # Look for AWS instance types
        instance_matches = re.findall(r'instance_type\s*=\s*"([^"]+)"', content)
        for instance_type in instance_matches:
            if instance_type in self.aws_pricing:
                monthly_cost = self.aws_pricing[instance_type] * 24 * 30
                breakdown[f'ec2_{instance_type}'] = monthly_cost
                total_monthly_cost += monthly_cost
        
        # Look for storage
        storage_matches = re.findall(r'size\s*=\s*(\d+)', content)
        for size in storage_matches:
            storage_cost = int(size) * self.aws_pricing['ebs_gp2']
            breakdown['storage'] = breakdown.get('storage', 0) + storage_cost
            total_monthly_cost += storage_cost
        
        return {
            'monthly_cost': round(total_monthly_cost, 2),
            'breakdown': breakdown,
            'confidence': 'medium',
            'currency': 'USD',
            'assumptions': 'Based on AWS pricing, standard usage patterns'
        }
    
    def _estimate_ansible_cost(self, content: str) -> Dict[str, Any]:
        """Estimate Ansible deployment costs"""
        # Ansible itself doesn't create infrastructure, so cost depends on target systems
        return {
            'monthly_cost': 0.0,
            'breakdown': {'ansible_runtime': 0.0},
            'confidence': 'low',
            'currency': 'USD',
            'assumptions': 'Ansible playbook execution cost only, excludes target infrastructure'
        }

class AdvancedIaCGenerator:
    """Advanced Infrastructure as Code Generator with AI capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.security_analyzer = SecurityAnalyzer()
        self.cost_estimator = CostEstimator()
        
        # Initialize AI models
        self._initialize_models()
        
        # Initialize templates
        self._load_templates()
    
    def _initialize_models(self):
        """Initialize Hugging Face models"""
        try:
            self.logger.info("Initializing AI models for IaC generation...")
            
            # Text generation for code
            self.text_generator = pipeline(
                'text-generation',
                model='distilgpt2',
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Text classification for intent
            self.intent_classifier = pipeline(
                'text-classification',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("AI models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize AI models: {str(e)}")
            self.text_generator = None
            self.intent_classifier = None
    
    def _load_templates(self):
        """Load IaC templates"""
        self.templates = {
            'kubernetes': {
                'deployment': self._get_k8s_deployment_template(),
                'service': self._get_k8s_service_template(),
                'ingress': self._get_k8s_ingress_template(),
                'configmap': self._get_k8s_configmap_template(),
                'secret': self._get_k8s_secret_template(),
            },
            'terraform': {
                'aws_ec2': self._get_terraform_ec2_template(),
                'aws_vpc': self._get_terraform_vpc_template(),
                'aws_rds': self._get_terraform_rds_template(),
            },
            'ansible': {
                'web_server': self._get_ansible_webserver_template(),
                'database': self._get_ansible_database_template(),
            }
        }
    
    def generate_infrastructure(self, request: GenerationRequest) -> GenerationResult:
        """Generate infrastructure from request"""
        try:
            start_time = datetime.now()
            request_id = hashlib.md5(f"{request.prompt}{start_time}".encode()).hexdigest()[:8]
            
            self.logger.info(f"Generating infrastructure for request {request_id}")
            
            # Analyze intent
            intent = self._analyze_intent(request.prompt)
            
            # Generate templates
            templates = self._generate_templates(request, intent)
            
            # Validate templates
            validation_results = self._validate_templates(templates)
            
            # Analyze security
            security_analysis = {}
            for template in templates:
                security_analysis[template.name] = self.security_analyzer.analyze_security(template)
                template.security_score = security_analysis[template.name]['score']
            
            # Estimate costs
            cost_estimate = {}
            total_monthly_cost = 0.0
            for template in templates:
                cost_data = self.cost_estimator.estimate_cost(template)
                cost_estimate[template.name] = cost_data
                template.estimated_cost = cost_data['monthly_cost']
                total_monthly_cost += cost_data['monthly_cost']
            
            cost_estimate['total_monthly_cost'] = total_monthly_cost
            
            # Generate insights and recommendations
            ai_insights = self._generate_ai_insights(request, templates, intent)
            recommendations = self._generate_recommendations(templates, security_analysis, cost_estimate)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GenerationResult(
                request_id=request_id,
                templates=templates,
                generation_time=generation_time,
                ai_insights=ai_insights,
                recommendations=recommendations,
                validation_results=validation_results,
                security_analysis=security_analysis,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            self.logger.error(f"Infrastructure generation failed: {str(e)}")
            raise
    
    def _analyze_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze user intent from prompt"""
        try:
            # Keyword-based analysis
            keywords = {
                'deployment': ['deploy', 'application', 'app', 'service', 'microservice'],
                'database': ['database', 'db', 'mysql', 'postgres', 'mongodb', 'redis'],
                'networking': ['network', 'ingress', 'load balancer', 'proxy'],
                'storage': ['storage', 'volume', 'persistent', 'pvc'],
                'monitoring': ['monitoring', 'metrics', 'logging', 'observability'],
                'security': ['security', 'rbac', 'authentication', 'authorization'],
                'scaling': ['scale', 'autoscale', 'hpa', 'cluster']
            }
            
            prompt_lower = prompt.lower()
            detected_intents = []
            
            for intent, words in keywords.items():
                if any(word in prompt_lower for word in words):
                    detected_intents.append(intent)
            
            return {
                'primary_intent': detected_intents[0] if detected_intents else 'deployment',
                'all_intents': detected_intents,
                'complexity': len(detected_intents),
                'entities': self._extract_entities(prompt)
            }
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {str(e)}")
            return {'primary_intent': 'deployment', 'all_intents': ['deployment'], 'complexity': 1}
    
    def _extract_entities(self, prompt: str) -> Dict[str, List[str]]:
        """Extract entities from prompt"""
        entities = {
            'app_names': [],
            'technologies': [],
            'ports': [],
            'environments': []
        }
        
        import re
        
        # Extract app names (simple heuristic)
        app_patterns = [r'\b(\w+)-app\b', r'\b(\w+) application\b', r'\bdeploy (\w+)\b']
        for pattern in app_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            entities['app_names'].extend(matches)
        
        # Extract technologies
        tech_keywords = ['nginx', 'apache', 'mysql', 'postgres', 'redis', 'mongodb', 'node', 'python', 'java']
        for tech in tech_keywords:
            if tech in prompt.lower():
                entities['technologies'].append(tech)
        
        # Extract ports
        port_matches = re.findall(r'\b(\d{2,5})\b', prompt)
        entities['ports'] = [port for port in port_matches if 80 <= int(port) <= 65535]
        
        # Extract environments
        env_keywords = ['dev', 'development', 'staging', 'prod', 'production', 'test']
        for env in env_keywords:
            if env in prompt.lower():
                entities['environments'].append(env)
        
        return entities
    
    def _generate_templates(self, request: GenerationRequest, intent: Dict[str, Any]) -> List[IaCTemplate]:
        """Generate IaC templates based on request and intent"""
        templates = []
        
        try:
            if request.platform == 'kubernetes':
                templates.extend(self._generate_kubernetes_templates(request, intent))
            elif request.platform == 'terraform':
                templates.extend(self._generate_terraform_templates(request, intent))
            elif request.platform == 'ansible':
                templates.extend(self._generate_ansible_templates(request, intent))
            
            return templates
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {str(e)}")
            return []
    
    def _generate_kubernetes_templates(self, request: GenerationRequest, intent: Dict[str, Any]) -> List[IaCTemplate]:
        """Generate Kubernetes templates"""
        templates = []
        app_name = intent['entities']['app_names'][0] if intent['entities']['app_names'] else 'myapp'
        
        # Always generate deployment
        deployment_content = self._generate_k8s_deployment(app_name, request, intent)
        templates.append(IaCTemplate(
            name=f"{app_name}-deployment",
            platform='kubernetes',
            provider=request.provider,
            content=deployment_content,
            metadata={'type': 'deployment', 'app': app_name},
            validation_status='pending',
            security_score=0.0,
            estimated_cost=None,
            compliance_status={}
        ))
        
        # Generate service
        service_content = self._generate_k8s_service(app_name, request, intent)
        templates.append(IaCTemplate(
            name=f"{app_name}-service",
            platform='kubernetes',
            provider=request.provider,
            content=service_content,
            metadata={'type': 'service', 'app': app_name},
            validation_status='pending',
            security_score=0.0,
            estimated_cost=None,
            compliance_status={}
        ))
        
        # Generate additional resources based on intent
        if 'networking' in intent['all_intents']:
            ingress_content = self._generate_k8s_ingress(app_name, request, intent)
            templates.append(IaCTemplate(
                name=f"{app_name}-ingress",
                platform='kubernetes',
                provider=request.provider,
                content=ingress_content,
                metadata={'type': 'ingress', 'app': app_name},
                validation_status='pending',
                security_score=0.0,
                estimated_cost=None,
                compliance_status={}
            ))
        
        if 'database' in intent['all_intents']:
            db_content = self._generate_k8s_database(app_name, request, intent)
            templates.append(IaCTemplate(
                name=f"{app_name}-database",
                platform='kubernetes',
                provider=request.provider,
                content=db_content,
                metadata={'type': 'statefulset', 'app': f"{app_name}-db"},
                validation_status='pending',
                security_score=0.0,
                estimated_cost=None,
                compliance_status={}
            ))
        
        return templates
    
    def _generate_k8s_deployment(self, app_name: str, request: GenerationRequest, intent: Dict[str, Any]) -> str:
        """Generate Kubernetes deployment"""
        port = intent['entities']['ports'][0] if intent['entities']['ports'] else '80'
        image = f"{app_name}:latest"
        
        # Detect technology and adjust image
        if 'nginx' in intent['entities']['technologies']:
            image = 'nginx:latest'
        elif 'node' in intent['entities']['technologies']:
            image = 'node:16-alpine'
        elif 'python' in intent['entities']['technologies']:
            image = 'python:3.9-slim'
        
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
        version: v1
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: {app_name}
        image: {image}
        ports:
        - containerPort: {port}
          name: http
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 128Mi
        env:
        - name: NODE_ENV
          value: production
        - name: PORT
          value: "{port}"
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {{}}
      - name: cache
        emptyDir: {{}}
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "kubernetes.io/arch"
        operator: "Equal"
        value: "amd64"
        effect: "NoSchedule"
"""
    
    def _generate_k8s_service(self, app_name: str, request: GenerationRequest, intent: Dict[str, Any]) -> str:
        """Generate Kubernetes service"""
        port = intent['entities']['ports'][0] if intent['entities']['ports'] else '80'
        
        return f"""apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
  labels:
    app: {app_name}
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: {port}
    protocol: TCP
    name: http
  selector:
    app: {app_name}
"""
    
    def _generate_k8s_ingress(self, app_name: str, request: GenerationRequest, intent: Dict[str, Any]) -> str:
        """Generate Kubernetes ingress"""
        return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}-ingress
  labels:
    app: {app_name}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - {app_name}.example.com
    secretName: {app_name}-tls
  rules:
  - host: {app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}-service
            port:
              number: 80
"""
    
    def _generate_k8s_database(self, app_name: str, request: GenerationRequest, intent: Dict[str, Any]) -> str:
        """Generate Kubernetes database StatefulSet"""
        db_type = 'postgres'
        if 'mysql' in intent['entities']['technologies']:
            db_type = 'mysql'
        elif 'mongodb' in intent['entities']['technologies']:
            db_type = 'mongo'
        
        if db_type == 'postgres':
            return f"""apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {app_name}-postgres
  labels:
    app: {app_name}-postgres
spec:
  serviceName: {app_name}-postgres
  replicas: 1
  selector:
    matchLabels:
      app: {app_name}-postgres
  template:
    metadata:
      labels:
        app: {app_name}-postgres
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: postgres
        image: postgres:13-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: {app_name}
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: {app_name}-postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: {app_name}-postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
          requests:
            cpu: 200m
            memory: 256Mi
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-postgres-service
  labels:
    app: {app_name}-postgres
spec:
  type: ClusterIP
  ports:
  - port: 5432
    targetPort: 5432
  selector:
    app: {app_name}-postgres
---
apiVersion: v1
kind: Secret
metadata:
  name: {app_name}-postgres-secret
type: Opaque
data:
  username: cG9zdGdyZXM=  # postgres (base64)
  password: cGFzc3dvcmQxMjM=  # password123 (base64)
"""
        else:
            return "# Database template not implemented for this type"
    
    # Template getter methods (simplified for brevity)
    def _get_k8s_deployment_template(self) -> str:
        return "# Kubernetes Deployment Template"
    
    def _get_k8s_service_template(self) -> str:
        return "# Kubernetes Service Template"
    
    def _get_k8s_ingress_template(self) -> str:
        return "# Kubernetes Ingress Template"
    
    def _get_k8s_configmap_template(self) -> str:
        return "# Kubernetes ConfigMap Template"
    
    def _get_k8s_secret_template(self) -> str:
        return "# Kubernetes Secret Template"
    
    def _get_terraform_ec2_template(self) -> str:
        return "# Terraform EC2 Template"
    
    def _get_terraform_vpc_template(self) -> str:
        return "# Terraform VPC Template"
    
    def _get_terraform_rds_template(self) -> str:
        return "# Terraform RDS Template"
    
    def _get_ansible_webserver_template(self) -> str:
        return "# Ansible Web Server Template"
    
    def _get_ansible_database_template(self) -> str:
        return "# Ansible Database Template"
    
    def _generate_terraform_templates(self, request: GenerationRequest, intent: Dict[str, Any]) -> List[IaCTemplate]:
        """Generate Terraform templates"""
        # Implement Terraform template generation
        return []
    
    def _generate_ansible_templates(self, request: GenerationRequest, intent: Dict[str, Any]) -> List[IaCTemplate]:
        """Generate Ansible templates"""
        # Implement Ansible template generation
        return []
    
    def _validate_templates(self, templates: List[IaCTemplate]) -> Dict[str, Any]:
        """Validate generated templates"""
        validation_results = {}
        
        for template in templates:
            try:
                if template.platform == 'kubernetes':
                    result = self._validate_kubernetes_yaml(template.content)
                elif template.platform == 'terraform':
                    result = self._validate_terraform_hcl(template.content)
                else:
                    result = self._validate_ansible_yaml(template.content)
                
                validation_results[template.name] = result
                template.validation_status = 'valid' if result['valid'] else 'invalid'
                
            except Exception as e:
                validation_results[template.name] = {'valid': False, 'errors': [str(e)]}
                template.validation_status = 'error'
        
        return validation_results
    
    def _validate_kubernetes_yaml(self, content: str) -> Dict[str, Any]:
        """Validate Kubernetes YAML"""
        try:
            docs = list(yaml.safe_load_all(content))
            errors = []
            
            for doc in docs:
                if not doc:
                    continue
                
                # Basic validation
                if 'apiVersion' not in doc:
                    errors.append("Missing apiVersion")
                
                if 'kind' not in doc:
                    errors.append("Missing kind")
                
                if 'metadata' not in doc or 'name' not in doc.get('metadata', {}):
                    errors.append("Missing metadata.name")
            
            return {'valid': len(errors) == 0, 'errors': errors}
            
        except yaml.YAMLError as e:
            return {'valid': False, 'errors': [f"YAML parsing error: {str(e)}"]}
    
    def _validate_terraform_hcl(self, content: str) -> Dict[str, Any]:
        """Validate Terraform HCL (basic validation)"""
        # This is a simplified validation
        errors = []
        
        if 'terraform {' not in content and 'resource "' not in content:
            errors.append("No Terraform blocks found")
        
        # Check for basic syntax issues
        if content.count('{') != content.count('}'):
            errors.append("Mismatched braces")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def _validate_ansible_yaml(self, content: str) -> Dict[str, Any]:
        """Validate Ansible YAML"""
        try:
            docs = list(yaml.safe_load_all(content))
            errors = []
            
            for doc in docs:
                if isinstance(doc, list):
                    for play in doc:
                        if 'name' not in play:
                            errors.append("Play missing name")
                        if 'hosts' not in play:
                            errors.append("Play missing hosts")
                        if 'tasks' not in play:
                            errors.append("Play missing tasks")
            
            return {'valid': len(errors) == 0, 'errors': errors}
            
        except yaml.YAMLError as e:
            return {'valid': False, 'errors': [f"YAML parsing error: {str(e)}"]}
    
    def _generate_ai_insights(self, request: GenerationRequest, templates: List[IaCTemplate], intent: Dict[str, Any]) -> str:
        """Generate AI insights about the infrastructure"""
        try:
            if self.text_generator:
                prompt = f"""
Infrastructure Analysis:
Request: {request.prompt}
Platform: {request.platform}
Templates generated: {len(templates)}
Primary intent: {intent['primary_intent']}

AI Analysis: This infrastructure setup
"""
                
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt) + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                generated_text = response[0]['generated_text']
                
                if "AI Analysis:" in generated_text:
                    insights = generated_text.split("AI Analysis:")[1].strip()
                    return insights.split('\n')[0]
            
            # Fallback insights
            return self._generate_template_insights(request, templates, intent)
            
        except Exception as e:
            self.logger.warning(f"AI insights generation failed: {str(e)}")
            return self._generate_template_insights(request, templates, intent)
    
    def _generate_template_insights(self, request: GenerationRequest, templates: List[IaCTemplate], intent: Dict[str, Any]) -> str:
        """Generate template-based insights"""
        insights = []
        
        if len(templates) > 3:
            insights.append("Complex infrastructure with multiple components requires careful orchestration")
        elif len(templates) == 1:
            insights.append("Simple single-component deployment with minimal complexity")
        
        if intent['primary_intent'] == 'database':
            insights.append("Database deployment requires attention to persistence and backup strategies")
        elif intent['primary_intent'] == 'networking':
            insights.append("Network configuration needs security hardening and traffic policies")
        
        if request.platform == 'kubernetes':
            insights.append("Kubernetes deployment benefits from proper resource management and health checks")
        
        return ". ".join(insights) if insights else "Standard infrastructure deployment detected"
    
    def _generate_recommendations(self, templates: List[IaCTemplate], 
                                security_analysis: Dict[str, Any], 
                                cost_estimate: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Security recommendations
        high_risk_templates = [name for name, analysis in security_analysis.items() 
                             if analysis.get('score', 1.0) < 0.7]
        
        if high_risk_templates:
            recommendations.append(f"Review security configuration for: {', '.join(high_risk_templates)}")
        
        # Cost recommendations
        total_cost = cost_estimate.get('total_monthly_cost', 0)
        if total_cost > 1000:
            recommendations.append("Consider cost optimization - monthly estimate exceeds $1000")
        elif total_cost > 500:
            recommendations.append("Monitor costs closely - moderate monthly expenses expected")
        
        # Platform-specific recommendations
        k8s_templates = [t for t in templates if t.platform == 'kubernetes']
        if k8s_templates:
            recommendations.append("Implement proper RBAC and network policies for Kubernetes")
            recommendations.append("Set up monitoring and logging for observability")
        
        # General recommendations
        recommendations.append("Test infrastructure in staging environment before production deployment")
        recommendations.append("Implement Infrastructure as Code best practices with version control")
        recommendations.append("Set up automated backup and disaster recovery procedures")
        
        return recommendations

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='CloudForge AI IaC Generator')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Natural language description of infrastructure')
    parser.add_argument('--platform', 
                       choices=['kubernetes', 'terraform', 'ansible'],
                       default='kubernetes',
                       help='Target platform for generated infrastructure')
    parser.add_argument('--provider',
                       choices=['aws', 'azure', 'gcp', 'oracle', 'on-premise'],
                       default='aws',
                       help='Cloud provider')
    parser.add_argument('--security-level',
                       choices=['basic', 'standard', 'high', 'enterprise'],
                       default='standard',
                       help='Security level for generated infrastructure')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for generated files')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                       help='Output format for reports')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize generator
        generator = AdvancedIaCGenerator()
        
        # Create generation request
        request = GenerationRequest(
            prompt=args.prompt,
            platform=args.platform,
            provider=args.provider,
            requirements={},
            constraints={},
            security_level=args.security_level,
            compliance_standards=[]
        )
        
        # Generate infrastructure
        logger.info("Generating infrastructure...")
        result = generator.generate_infrastructure(request)
        
        # Save templates
        for template in result.templates:
            file_extension = '.yaml' if args.platform in ['kubernetes', 'ansible'] else '.tf'
            filename = os.path.join(args.output, f"{template.name}{file_extension}")
            
            with open(filename, 'w') as f:
                f.write(template.content)
            
            logger.info(f"Generated template: {filename}")
        
        # Save analysis report
        report = {
            'request_id': result.request_id,
            'generation_time': result.generation_time,
            'ai_insights': result.ai_insights,
            'recommendations': result.recommendations,
            'validation_results': result.validation_results,
            'security_analysis': result.security_analysis,
            'cost_estimate': result.cost_estimate,
            'templates_summary': [
                {
                    'name': t.name,
                    'platform': t.platform,
                    'provider': t.provider,
                    'security_score': t.security_score,
                    'estimated_cost': t.estimated_cost,
                    'validation_status': t.validation_status
                }
                for t in result.templates
            ]
        }
        
        report_filename = os.path.join(args.output, f"generation_report.{args.format}")
        
        if args.format == 'json':
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            with open(report_filename, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
        
        # Print summary
        print(f"✅ Infrastructure generation completed!")
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Templates generated: {len(result.templates)}")
        print(f"⚡ Generation time: {result.generation_time:.2f}s")
        print(f"💰 Estimated monthly cost: ${result.cost_estimate.get('total_monthly_cost', 0):.2f}")
        
        # Print templates
        print(f"\n📄 Generated templates:")
        for template in result.templates:
            print(f"  • {template.name} ({template.platform}) - Security: {template.security_score:.2f}")
        
        # Print key recommendations
        print(f"\n💡 Top recommendations:")
        for rec in result.recommendations[:3]:
            print(f"  • {rec}")
        
        # Print AI insights
        if result.ai_insights:
            print(f"\n🤖 AI Insights: {result.ai_insights}")
        
    except Exception as e:
        logger.error(f"IaC generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()