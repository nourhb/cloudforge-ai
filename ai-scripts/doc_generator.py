#!/usr/bin/env python3
"""
CloudForge AI - Documentation Generator
Generates comprehensive documentation using Hugging Face NLP models.

This module provides intelligent documentation generation capabilities:
- API documentation generation from code analysis
- User guide creation with natural language generation
- Architecture documentation with system analysis
- PDF generation with proper formatting

Dependencies:
- transformers: Hugging Face transformers for NLP
- torch: PyTorch for model execution
- reportlab: PDF generation
- markdown: Markdown processing
- ast: Python AST parsing
- jinja2: Template rendering

Usage:
    python doc_generator.py --project /path/to/project --output docs/
    
    from doc_generator import DocumentationGenerator
    generator = DocumentationGenerator('/path/to/project')
    generator.generate_all_docs('output_directory')
"""

import os
import sys
import ast
import json
import logging
import argparse
import inspect
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import markdown
from jinja2 import Template, Environment, FileSystemLoader
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DistilBertTokenizer,
    DistilBertModel,
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/doc_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """Data structure for API endpoint documentation"""
    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    authentication: bool
    tags: List[str]

@dataclass
class ModuleDocumentation:
    """Data structure for module documentation"""
    name: str
    description: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    dependencies: List[str]
    examples: List[str]

@dataclass
class ArchitectureComponent:
    """Data structure for architecture component"""
    name: str
    type: str  # service, database, cache, etc.
    description: str
    dependencies: List[str]
    technologies: List[str]
    scaling_info: str
    responsibilities: List[str]

class NLPDocumentationEngine:
    """AI-powered documentation generation engine using Hugging Face models"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for documentation generation"""
        try:
            # Load DistilGPT2 for text generation
            self.logger.info("Loading DistilGPT2 model for text generation...")
            self.gpt_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            self.gpt_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
            
            # Add pad token if it doesn't exist
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            # Load DistilBERT for text understanding
            self.logger.info("Loading DistilBERT model for text analysis...")
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
            # Initialize text generation pipeline
            self.text_generator = pipeline(
                'text-generation',
                model=self.gpt_model,
                tokenizer=self.gpt_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {str(e)}")
            raise
    
    def generate_description(self, context: str, doc_type: str = "general") -> str:
        """Generate natural language description using AI"""
        try:
            # Create appropriate prompt based on documentation type
            if doc_type == "api":
                prompt = f"""
                Generate a clear and professional API documentation description for:
                
                Context: {context}
                
                The description should be:
                - Clear and concise
                - Technical but accessible
                - Include purpose and usage
                - Professional tone
                
                Description:
                """
            elif doc_type == "architecture":
                prompt = f"""
                Generate a comprehensive architecture description for:
                
                Context: {context}
                
                The description should explain:
                - Component purpose and role
                - How it fits in the overall system
                - Key technologies and design decisions
                - Scalability and performance considerations
                
                Description:
                """
            elif doc_type == "user_guide":
                prompt = f"""
                Generate a user-friendly guide description for:
                
                Context: {context}
                
                The description should be:
                - Easy to understand for end users
                - Step-by-step where applicable
                - Include practical examples
                - Friendly and helpful tone
                
                Description:
                """
            else:
                prompt = f"""
                Generate a clear technical description for:
                
                Context: {context}
                
                Description:
                """
            
            # Generate description using AI
            try:
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt) + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.gpt_tokenizer.eos_token_id
                )
                
                generated_text = response[0]['generated_text']
                
                # Extract the description part
                description = self._extract_description(generated_text, prompt)
                return description
                
            except Exception as e:
                self.logger.warning(f"AI generation failed, using template: {str(e)}")
                return self._generate_template_description(context, doc_type)
            
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return f"Auto-generated description for {context}"
    
    def _extract_description(self, generated_text: str, prompt: str) -> str:
        """Extract description from AI-generated text"""
        try:
            # Find the description part after the prompt
            if "Description:" in generated_text:
                parts = generated_text.split("Description:", 1)
                if len(parts) > 1:
                    description = parts[1].strip()
                    # Clean up the description
                    lines = description.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('Context:') and not line.startswith('The description'):
                            clean_lines.append(line)
                            if len(' '.join(clean_lines)) > 200:  # Reasonable length
                                break
                    
                    if clean_lines:
                        return ' '.join(clean_lines)
            
            # Fallback to template description
            return "AI-generated documentation description"
            
        except Exception as e:
            self.logger.error(f"Error extracting description: {str(e)}")
            return "Documentation description"
    
    def _generate_template_description(self, context: str, doc_type: str) -> str:
        """Generate template-based description as fallback"""
        if doc_type == "api":
            return f"This API endpoint provides functionality for {context}. It handles requests and returns appropriate responses based on the input parameters."
        elif doc_type == "architecture":
            return f"This component ({context}) is a critical part of the CloudForge AI system architecture, providing essential functionality for the overall system operation."
        elif doc_type == "user_guide":
            return f"This section explains how to use {context} effectively. Follow the steps and examples provided to get the most out of this feature."
        else:
            return f"This section covers {context} and its implementation details."

class CodeAnalyzer:
    """Analyzes source code to extract documentation information"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_python_files(self) -> Dict[str, ModuleDocumentation]:
        """Analyze Python files and extract documentation information"""
        try:
            modules = {}
            
            # Find all Python files
            python_files = list(self.project_path.rglob("*.py"))
            self.logger.info(f"Found {len(python_files)} Python files to analyze")
            
            for py_file in python_files:
                if py_file.name.startswith('__'):
                    continue
                
                try:
                    module_doc = self._analyze_python_module(py_file)
                    if module_doc:
                        modules[py_file.stem] = module_doc
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {py_file}: {str(e)}")
            
            return modules
            
        except Exception as e:
            self.logger.error(f"Error analyzing Python files: {str(e)}")
            return {}
    
    def _analyze_python_module(self, file_path: Path) -> Optional[ModuleDocumentation]:
        """Analyze a single Python module"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse AST
            tree = ast.parse(source)
            
            # Extract module docstring
            description = ast.get_docstring(tree) or f"Module: {file_path.stem}"
            
            # Extract classes
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node)
                    classes.append(class_info)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    func_info = self._extract_function_info(node)
                    functions.append(func_info)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return ModuleDocumentation(
                name=file_path.stem,
                description=description,
                classes=classes,
                functions=functions,
                dependencies=list(set(imports)),
                examples=self._generate_code_examples(classes, functions)
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing module {file_path}: {str(e)}")
            return None
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information from AST node"""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item)
                methods.append(method_info)
        
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or f"Class {node.name}",
            'methods': methods,
            'base_classes': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function information from AST node"""
        args = []
        
        for arg in node.args.args:
            args.append({
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None
            })
        
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or f"Function {node.name}",
            'arguments': args,
            'returns': ast.unparse(node.returns) if node.returns else None,
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _generate_code_examples(self, classes: List[Dict], functions: List[Dict]) -> List[str]:
        """Generate code usage examples"""
        examples = []
        
        # Generate class usage examples
        for cls in classes[:2]:  # Limit to first 2 classes
            example = f"""
# Example usage of {cls['name']}
{cls['name'].lower()} = {cls['name']}()
result = {cls['name'].lower()}.{cls['methods'][0]['name']}() if cls['methods'] else 'method()'
print(result)
"""
            examples.append(example.strip())
        
        # Generate function usage examples
        for func in functions[:2]:  # Limit to first 2 functions
            args = ', '.join([arg['name'] for arg in func['arguments']])
            example = f"""
# Example usage of {func['name']}
result = {func['name']}({args})
print(result)
"""
            examples.append(example.strip())
        
        return examples
    
    def analyze_api_endpoints(self) -> List[APIEndpoint]:
        """Analyze API endpoints from NestJS controllers"""
        try:
            endpoints = []
            
            # Find TypeScript controller files
            controller_files = list(self.project_path.rglob("*.controller.ts"))
            self.logger.info(f"Found {len(controller_files)} controller files")
            
            for controller_file in controller_files:
                try:
                    controller_endpoints = self._analyze_controller_file(controller_file)
                    endpoints.extend(controller_endpoints)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze controller {controller_file}: {str(e)}")
            
            return endpoints
            
        except Exception as e:
            self.logger.error(f"Error analyzing API endpoints: {str(e)}")
            return []
    
    def _analyze_controller_file(self, file_path: Path) -> List[APIEndpoint]:
        """Analyze a single controller file for API endpoints"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            endpoints = []
            
            # Extract basic endpoint information (simplified parsing)
            lines = content.split('\n')
            current_method = None
            current_path = None
            
            for line in lines:
                line = line.strip()
                
                # Look for HTTP method decorators
                if line.startswith('@Get(') or line.startswith('@Post(') or \
                   line.startswith('@Put(') or line.startswith('@Delete('):
                    method = line.split('(')[0][1:]  # Remove @ and get method
                    path_match = line.split("'")
                    current_path = path_match[1] if len(path_match) > 1 else '/'
                    current_method = method.upper()
                
                # Look for function definitions
                elif current_method and line.startswith('async ') or line.startswith('public '):
                    if '(' in line and ')' in line:
                        func_name = line.split('(')[0].split()[-1]
                        
                        endpoint = APIEndpoint(
                            path=current_path or f"/{func_name}",
                            method=current_method,
                            description=f"{current_method} endpoint for {func_name}",
                            parameters=[],
                            responses=[
                                {'status': 200, 'description': 'Success'},
                                {'status': 400, 'description': 'Bad Request'}
                            ],
                            examples=[],
                            authentication=True,
                            tags=[file_path.stem.replace('.controller', '')]
                        )
                        
                        endpoints.append(endpoint)
                        current_method = None
                        current_path = None
            
            return endpoints
            
        except Exception as e:
            self.logger.error(f"Error analyzing controller {file_path}: {str(e)}")
            return []

class DocumentationGenerator:
    """Main documentation generator class"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nlp_engine = NLPDocumentationEngine()
        self.code_analyzer = CodeAnalyzer(project_path)
        
        # Create Jinja2 environment for templates
        template_dir = Path(__file__).parent / 'templates'
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
    
    def generate_all_docs(self, output_dir: str):
        """Generate all documentation types"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info("Starting comprehensive documentation generation...")
        
        # Generate API documentation
        self.generate_api_documentation(output_path / 'api.md')
        
        # Generate user guide PDF
        self.generate_user_guide_pdf(output_path / 'user_guide.pdf')
        
        # Generate architecture documentation
        self.generate_architecture_documentation(output_path / 'architecture.md')
        
        # Generate additional documentation files
        self.generate_installation_guide(output_path / 'installation.md')
        self.generate_deployment_guide(output_path / 'deployment.md')
        
        self.logger.info("Documentation generation completed successfully")
    
    def generate_api_documentation(self, output_file: Path):
        """Generate comprehensive API documentation"""
        try:
            self.logger.info("Generating API documentation...")
            
            # Analyze API endpoints
            endpoints = self.code_analyzer.analyze_api_endpoints()
            
            # Generate content using templates and AI
            api_content = self._generate_api_content(endpoints)
            
            # Write to markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(api_content)
            
            self.logger.info(f"API documentation generated: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating API documentation: {str(e)}")
            raise
    
    def _generate_api_content(self, endpoints: List[APIEndpoint]) -> str:
        """Generate API documentation content"""
        content = []
        
        # Header
        content.append("# CloudForge AI - API Documentation")
        content.append("")
        content.append(self.nlp_engine.generate_description(
            "CloudForge AI REST API", "api"
        ))
        content.append("")
        content.append("## Table of Contents")
        content.append("")
        
        # Group endpoints by tags
        endpoint_groups = {}
        for endpoint in endpoints:
            tag = endpoint.tags[0] if endpoint.tags else 'general'
            if tag not in endpoint_groups:
                endpoint_groups[tag] = []
            endpoint_groups[tag].append(endpoint)
        
        # Generate table of contents
        for tag in sorted(endpoint_groups.keys()):
            content.append(f"- [{tag.title()}](#{tag.lower()})")
        content.append("")
        
        # Generate endpoint documentation
        for tag, tag_endpoints in endpoint_groups.items():
            content.append(f"## {tag.title()}")
            content.append("")
            content.append(self.nlp_engine.generate_description(
                f"{tag} API endpoints", "api"
            ))
            content.append("")
            
            for endpoint in tag_endpoints:
                content.extend(self._format_endpoint_documentation(endpoint))
        
        return '\n'.join(content)
    
    def _format_endpoint_documentation(self, endpoint: APIEndpoint) -> List[str]:
        """Format individual endpoint documentation"""
        content = []
        
        content.append(f"### {endpoint.method} {endpoint.path}")
        content.append("")
        content.append(endpoint.description)
        content.append("")
        
        # Parameters
        if endpoint.parameters:
            content.append("**Parameters:**")
            content.append("")
            for param in endpoint.parameters:
                content.append(f"- `{param['name']}` ({param.get('type', 'string')}): {param.get('description', 'Parameter description')}")
            content.append("")
        
        # Responses
        content.append("**Responses:**")
        content.append("")
        for response in endpoint.responses:
            content.append(f"- `{response['status']}`: {response['description']}")
        content.append("")
        
        # Example
        content.append("**Example Request:**")
        content.append("")
        content.append("```bash")
        content.append(f"curl -X {endpoint.method} \\")
        content.append(f"  'https://api.cloudforge.com{endpoint.path}' \\")
        if endpoint.authentication:
            content.append("  -H 'Authorization: Bearer YOUR_TOKEN' \\")
        content.append("  -H 'Content-Type: application/json'")
        content.append("```")
        content.append("")
        
        return content
    
    def generate_user_guide_pdf(self, output_file: Path):
        """Generate user guide as PDF"""
        try:
            self.logger.info("Generating user guide PDF...")
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_file), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("CloudForge AI User Guide", title_style))
            story.append(Spacer(1, 20))
            
            # Introduction
            intro_text = self.nlp_engine.generate_description(
                "CloudForge AI platform for cloud migration and optimization", "user_guide"
            )
            story.append(Paragraph("Introduction", styles['Heading2']))
            story.append(Paragraph(intro_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Getting Started
            story.append(Paragraph("Getting Started", styles['Heading2']))
            getting_started_text = """
            CloudForge AI is a comprehensive platform for cloud migration, optimization, and management. 
            This guide will help you get started with the platform and make the most of its features.
            """
            story.append(Paragraph(getting_started_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Features sections
            features = [
                ("Database Migration", "Learn how to migrate your databases using AI-powered analysis"),
                ("Marketplace Analytics", "Discover how to analyze and optimize your marketplace performance"),
                ("Infrastructure as Code", "Generate and manage infrastructure templates"),
                ("AI-Powered Insights", "Leverage machine learning for optimization recommendations")
            ]
            
            for feature_title, feature_desc in features:
                story.append(Paragraph(feature_title, styles['Heading3']))
                enhanced_desc = self.nlp_engine.generate_description(feature_desc, "user_guide")
                story.append(Paragraph(enhanced_desc, styles['Normal']))
                story.append(Spacer(1, 12))
            
            # FAQ Section
            story.append(PageBreak())
            story.append(Paragraph("Frequently Asked Questions", styles['Heading2']))
            
            faqs = [
                ("How do I start a migration?", "Navigate to the Migration section and follow the step-by-step wizard."),
                ("What databases are supported?", "CloudForge AI supports PostgreSQL, MySQL, SQLite, and more."),
                ("How do I access API documentation?", "API documentation is available in the docs section of the platform."),
                ("What are the system requirements?", "CloudForge AI is cloud-based and accessible via web browser.")
            ]
            
            for question, answer in faqs:
                story.append(Paragraph(f"Q: {question}", styles['Heading4']))
                story.append(Paragraph(f"A: {answer}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"User guide PDF generated: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating user guide PDF: {str(e)}")
            raise
    
    def generate_architecture_documentation(self, output_file: Path):
        """Generate architecture documentation"""
        try:
            self.logger.info("Generating architecture documentation...")
            
            # Define architecture components
            components = [
                ArchitectureComponent(
                    name="Backend API",
                    type="service",
                    description="NestJS-based REST API service",
                    dependencies=["PostgreSQL", "Redis", "AI Services"],
                    technologies=["Node.js", "TypeScript", "NestJS"],
                    scaling_info="Horizontally scalable with load balancer",
                    responsibilities=["API endpoints", "Business logic", "Authentication"]
                ),
                ArchitectureComponent(
                    name="Frontend Application",
                    type="service",
                    description="Next.js React application",
                    dependencies=["Backend API"],
                    technologies=["React", "Next.js", "TypeScript", "Tailwind CSS"],
                    scaling_info="CDN deployment with edge caching",
                    responsibilities=["User interface", "Client-side logic", "State management"]
                ),
                ArchitectureComponent(
                    name="AI Services",
                    type="service",
                    description="Python-based AI/ML processing services",
                    dependencies=["PostgreSQL", "File Storage"],
                    technologies=["Python", "Hugging Face", "PyTorch", "scikit-learn"],
                    scaling_info="Auto-scaling based on processing queue",
                    responsibilities=["ML processing", "Anomaly detection", "Optimization"]
                ),
                ArchitectureComponent(
                    name="PostgreSQL Database",
                    type="database",
                    description="Primary relational database",
                    dependencies=[],
                    technologies=["PostgreSQL 14", "Connection pooling"],
                    scaling_info="Read replicas and partitioning",
                    responsibilities=["Data persistence", "ACID transactions", "Relational data"]
                ),
                ArchitectureComponent(
                    name="Redis Cache",
                    type="cache",
                    description="In-memory caching and session storage",
                    dependencies=[],
                    technologies=["Redis 7", "Clustering"],
                    scaling_info="Cluster mode with automatic failover",
                    responsibilities=["Session storage", "API caching", "Real-time data"]
                )
            ]
            
            # Generate architecture content
            content = self._generate_architecture_content(components)
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Architecture documentation generated: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating architecture documentation: {str(e)}")
            raise
    
    def _generate_architecture_content(self, components: List[ArchitectureComponent]) -> str:
        """Generate architecture documentation content"""
        content = []
        
        # Header
        content.append("# CloudForge AI - System Architecture")
        content.append("")
        content.append(self.nlp_engine.generate_description(
            "CloudForge AI system architecture and component design", "architecture"
        ))
        content.append("")
        
        # Overview
        content.append("## Architecture Overview")
        content.append("")
        content.append("CloudForge AI follows a modern microservices architecture with the following key characteristics:")
        content.append("")
        content.append("- **Scalable**: Components can be scaled independently")
        content.append("- **Resilient**: Fault tolerance and graceful degradation")
        content.append("- **Maintainable**: Clear separation of concerns")
        content.append("- **Observable**: Comprehensive monitoring and logging")
        content.append("")
        
        # Component Details
        content.append("## System Components")
        content.append("")
        
        for component in components:
            content.append(f"### {component.name}")
            content.append("")
            
            # Enhanced description using AI
            enhanced_desc = self.nlp_engine.generate_description(
                f"{component.name}: {component.description}", "architecture"
            )
            content.append(enhanced_desc)
            content.append("")
            
            content.append(f"**Type:** {component.type}")
            content.append(f"**Technologies:** {', '.join(component.technologies)}")
            content.append(f"**Scaling:** {component.scaling_info}")
            content.append("")
            
            content.append("**Key Responsibilities:**")
            for responsibility in component.responsibilities:
                content.append(f"- {responsibility}")
            content.append("")
            
            if component.dependencies:
                content.append("**Dependencies:**")
                for dependency in component.dependencies:
                    content.append(f"- {dependency}")
                content.append("")
        
        # Data Flow
        content.append("## Data Flow")
        content.append("")
        content.append("1. **User Request**: Frontend sends request to Backend API")
        content.append("2. **Authentication**: JWT token validation via Redis")
        content.append("3. **Business Logic**: Backend processes request")
        content.append("4. **Data Access**: PostgreSQL for persistent data")
        content.append("5. **AI Processing**: Python services for ML operations")
        content.append("6. **Response**: JSON response back to frontend")
        content.append("")
        
        # Deployment
        content.append("## Deployment Architecture")
        content.append("")
        content.append("CloudForge AI is designed for cloud-native deployment:")
        content.append("")
        content.append("- **Containerization**: Docker containers for all services")
        content.append("- **Orchestration**: Kubernetes for container management")
        content.append("- **Load Balancing**: NGINX/HAProxy for traffic distribution")
        content.append("- **Monitoring**: Prometheus and Grafana for observability")
        content.append("- **CI/CD**: GitHub Actions for automated deployment")
        content.append("")
        
        return '\n'.join(content)
    
    def generate_installation_guide(self, output_file: Path):
        """Generate installation guide"""
        content = [
            "# CloudForge AI - Installation Guide",
            "",
            "This guide covers the installation and setup of CloudForge AI platform.",
            "",
            "## Prerequisites",
            "",
            "- Node.js 18+ (for backend and frontend)",
            "- Python 3.11+ (for AI services)",
            "- PostgreSQL 14+ (database)",
            "- Redis 7+ (caching)",
            "- Docker & Docker Compose (for containerized deployment)",
            "",
            "## Quick Start with Docker",
            "",
            "```bash",
            "# Clone the repository",
            "git clone https://github.com/your-org/cloudforge-ai.git",
            "cd cloudforge-ai",
            "",
            "# Start all services",
            "docker-compose up -d",
            "",
            "# Access the application",
            "# Frontend: http://localhost:3002",
            "# Backend API: http://localhost:3001",
            "```",
            "",
            "## Manual Installation",
            "",
            "### Backend Setup",
            "",
            "```bash",
            "cd backend",
            "npm install",
            "cp .env.example .env",
            "# Edit .env with your configuration",
            "npm run build",
            "npm start",
            "```",
            "",
            "### Frontend Setup",
            "",
            "```bash",
            "cd frontend",
            "npm install",
            "cp .env.local.example .env.local",
            "# Edit .env.local with your configuration",
            "npm run build",
            "npm start",
            "```",
            "",
            "### AI Services Setup",
            "",
            "```bash",
            "cd ai-scripts",
            "pip install -r requirements.txt",
            "python app.py",
            "```"
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
    
    def generate_deployment_guide(self, output_file: Path):
        """Generate deployment guide"""
        content = [
            "# CloudForge AI - Deployment Guide",
            "",
            "Production deployment guide for CloudForge AI platform.",
            "",
            "## Kubernetes Deployment",
            "",
            "```bash",
            "# Apply Kubernetes manifests",
            "kubectl apply -f k8s/production/",
            "",
            "# Verify deployment",
            "kubectl get pods -n cloudforge-prod",
            "```",
            "",
            "## Environment Configuration",
            "",
            "### Production Environment Variables",
            "",
            "```bash",
            "# Backend",
            "NODE_ENV=production",
            "DATABASE_URL=postgresql://...",
            "REDIS_URL=redis://...",
            "JWT_SECRET=your-secret-key",
            "",
            "# Frontend",
            "NEXT_PUBLIC_API_URL=https://api.your-domain.com",
            "```",
            "",
            "## Monitoring Setup",
            "",
            "```bash",
            "# Deploy monitoring stack",
            "kubectl apply -f k8s/monitoring/",
            "",
            "# Access Grafana",
            "kubectl port-forward svc/grafana 3000:3000",
            "```",
            "",
            "## SSL/TLS Configuration",
            "",
            "```yaml",
            "# cert-manager for automatic SSL",
            "apiVersion: cert-manager.io/v1",
            "kind: ClusterIssuer",
            "metadata:",
            "  name: letsencrypt-prod",
            "spec:",
            "  acme:",
            "    server: https://acme-v02.api.letsencrypt.org/directory",
            "    email: admin@your-domain.com",
            "```"
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='CloudForge AI Documentation Generator')
    parser.add_argument('--project', default='.', 
                       help='Project root directory (default: current directory)')
    parser.add_argument('--output', default='docs',
                       help='Output directory for generated documentation')
    parser.add_argument('--type', choices=['all', 'api', 'user-guide', 'architecture'],
                       default='all', help='Type of documentation to generate')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create logs and output directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize generator
        generator = DocumentationGenerator(args.project)
        
        # Generate documentation
        if args.type == 'all':
            generator.generate_all_docs(args.output)
        elif args.type == 'api':
            generator.generate_api_documentation(Path(args.output) / 'api.md')
        elif args.type == 'user-guide':
            generator.generate_user_guide_pdf(Path(args.output) / 'user_guide.pdf')
        elif args.type == 'architecture':
            generator.generate_architecture_documentation(Path(args.output) / 'architecture.md')
        
        print(f"✅ Documentation generation completed successfully!")
        print(f"📁 Output directory: {args.output}")
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()