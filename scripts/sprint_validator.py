# scripts/sprint_validator.py - Development Sprint Validation Tool
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class SprintValidator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.validation_results = {}
        
    def validate_sprint_1(self):
        """Sprint 1: Project Foundation and Architecture"""
        print("🚀 Validating Sprint 1: Project Foundation")
        
        results = {}
        
        # Check README.md
        readme_path = self.project_root / "README.md"
        results['readme'] = {
            'exists': readme_path.exists(),
            'size': readme_path.stat().st_size if readme_path.exists() else 0
        }
        
        # Check docker-compose.yml
        compose_path = self.project_root / "docker-compose.yml"
        results['docker_compose'] = {
            'exists': compose_path.exists(),
            'services': self.count_docker_services(compose_path) if compose_path.exists() else 0
        }
        
        # Check directory structure
        required_dirs = ['frontend', 'backend', 'ai-scripts', 'infra', 'tests']
        results['directory_structure'] = {
            dir_name: (self.project_root / dir_name).exists() 
            for dir_name in required_dirs
        }
        
        # Validate package.json files
        frontend_pkg = self.project_root / "frontend" / "package.json"
        backend_pkg = self.project_root / "backend" / "package.json"
        results['package_files'] = {
            'frontend': frontend_pkg.exists(),
            'backend': backend_pkg.exists()
        }
        
        self.validation_results['sprint_1'] = results
        return results
        
    def validate_sprint_2(self):
        """Sprint 2: Core Infrastructure"""
        print("🏗️ Validating Sprint 2: Core Infrastructure")
        
        results = {}
        
        # Check Kubernetes manifests
        k8s_path = self.project_root / "infra" / "k8s-manifests"
        if k8s_path.exists():
            k8s_files = list(k8s_path.glob("*.yml"))
            results['k8s_manifests'] = {
                'count': len(k8s_files),
                'files': [f.name for f in k8s_files]
            }
        else:
            results['k8s_manifests'] = {'count': 0, 'files': []}
        
        # Check Helm chart
        helm_path = self.project_root / "helm-chart"
        results['helm_chart'] = {
            'exists': helm_path.exists(),
            'chart_yaml': (helm_path / "Chart.yaml").exists() if helm_path.exists() else False,
            'values_yaml': (helm_path / "values.yaml").exists() if helm_path.exists() else False
        }
        
        # Check monitoring setup
        monitoring_dirs = ['prometheus', 'grafana', 'airflow']
        results['monitoring'] = {
            dir_name: (self.project_root / "infra" / dir_name).exists()
            for dir_name in monitoring_dirs
        }
        
        self.validation_results['sprint_2'] = results
        return results
        
    def validate_sprint_3(self):
        """Sprint 3: Frontend Development"""
        print("🎨 Validating Sprint 3: Frontend Development")
        
        results = {}
        
        frontend_path = self.project_root / "frontend"
        
        # Check Next.js configuration
        results['nextjs_config'] = {
            'next_config': (frontend_path / "next.config.js").exists(),
            'tsconfig': (frontend_path / "tsconfig.json").exists(),
            'tailwind_config': (frontend_path / "tailwind.config.js").exists()
        }
        
        # Check component structure
        components_path = frontend_path / "src" / "components"
        if components_path.exists():
            component_dirs = [d for d in components_path.iterdir() if d.is_dir()]
            results['components'] = {
                'count': len(component_dirs),
                'directories': [d.name for d in component_dirs]
            }
        else:
            results['components'] = {'count': 0, 'directories': []}
        
        # Check app pages
        app_path = frontend_path / "src" / "app"
        if app_path.exists():
            page_files = list(app_path.rglob("page.tsx"))
            results['pages'] = {
                'count': len(page_files),
                'files': [str(f.relative_to(app_path)) for f in page_files]
            }
        else:
            results['pages'] = {'count': 0, 'files': []}
        
        # Check Cypress tests
        cypress_path = frontend_path / "cypress"
        results['cypress_tests'] = {
            'exists': cypress_path.exists(),
            'e2e_tests': len(list((cypress_path / "e2e").glob("*.cy.ts"))) if cypress_path.exists() else 0
        }
        
        self.validation_results['sprint_3'] = results
        return results
        
    def validate_sprint_4(self):
        """Sprint 4: Backend API Development"""
        print("⚙️ Validating Sprint 4: Backend Development")
        
        results = {}
        
        backend_path = self.project_root / "backend"
        
        # Check NestJS structure
        src_path = backend_path / "src"
        results['nestjs_structure'] = {
            'app_module': (src_path / "app.module.ts").exists(),
            'main_ts': (src_path / "main.ts").exists(),
            'modules_dir': (src_path / "modules").exists()
        }
        
        # Check API modules
        modules_path = src_path / "modules"
        if modules_path.exists():
            module_dirs = [d for d in modules_path.iterdir() if d.is_dir()]
            results['api_modules'] = {
                'count': len(module_dirs),
                'modules': [d.name for d in module_dirs]
            }
        else:
            results['api_modules'] = {'count': 0, 'modules': []}
        
        # Check controllers and services
        controllers = list(src_path.rglob("*.controller.ts"))
        services = list(src_path.rglob("*.service.ts"))
        results['api_files'] = {
            'controllers': len(controllers),
            'services': len(services)
        }
        
        # Check tests
        test_path = backend_path / "test"
        results['backend_tests'] = {
            'exists': test_path.exists(),
            'e2e_tests': len(list(test_path.glob("*.e2e-spec.ts"))) if test_path.exists() else 0
        }
        
        self.validation_results['sprint_4'] = results
        return results
        
    def validate_sprint_5(self):
        """Sprint 5: AI Services Integration"""
        print("🤖 Validating Sprint 5: AI Services")
        
        results = {}
        
        ai_path = self.project_root / "ai-scripts"
        
        # Check AI service files
        ai_services = [
            'anomaly_detector.py',
            'forecasting.py', 
            'iac_generator.py',
            'migration_analyzer.py',
            'doc_generator.py'
        ]
        
        results['ai_services'] = {
            service: (ai_path / service).exists()
            for service in ai_services
        }
        
        # Check requirements.txt
        requirements_path = ai_path / "requirements.txt"
        results['ai_requirements'] = {
            'exists': requirements_path.exists(),
            'packages': self.count_requirements(requirements_path) if requirements_path.exists() else 0
        }
        
        # Check Flask app
        results['flask_app'] = {
            'app_py': (ai_path / "app.py").exists(),
            'dockerfile': (ai_path / "Dockerfile").exists()
        }
        
        self.validation_results['sprint_5'] = results
        return results
        
    def validate_sprint_6(self):
        """Sprint 6: Database Integration"""
        print("💾 Validating Sprint 6: Database Integration")
        
        results = {}
        
        # Check database configurations
        infra_path = self.project_root / "infra"
        
        # MongoDB setup
        mongodb_path = infra_path / "mongodb"
        results['mongodb'] = {
            'exists': mongodb_path.exists(),
            'init_js': (mongodb_path / "init.js").exists() if mongodb_path.exists() else False
        }
        
        # Check database manifests
        db_manifests = ['mysql-deployment.yml', 'postgresql-deployment.yml']
        results['db_manifests'] = {
            manifest: (infra_path / "k8s-manifests" / manifest).exists()
            for manifest in db_manifests
        }
        
        # Check real datasets
        data_path = self.project_root / "data"
        results['datasets'] = {
            'chinook_db': (data_path / "chinook" / "chinook.db").exists() if data_path.exists() else False,
            'uci_network': (data_path / "uci-network").exists() if data_path.exists() else False,
            'kaggle_ecommerce': (data_path / "kaggle-ecommerce").exists() if data_path.exists() else False
        }
        
        self.validation_results['sprint_6'] = results
        return results
        
    def validate_sprint_7_to_12(self):
        """Validate remaining sprints (7-12)"""
        
        # Sprint 7: Security Implementation
        print("🔒 Validating Sprint 7: Security")
        security_results = {
            'jwt_service': (self.project_root / "backend" / "src" / "security" / "jwt.service.ts").exists(),
            'jwt_guard': (self.project_root / "backend" / "src" / "security" / "jwt.guard.ts").exists(),
            'security_tests': (self.project_root / "tests" / "security").exists()
        }
        self.validation_results['sprint_7'] = security_results
        
        # Sprint 8: Performance Optimization  
        print("⚡ Validating Sprint 8: Performance")
        perf_results = {
            'perf_tests': (self.project_root / "tests" / "perf").exists(),
            'perf_stats': (self.project_root / "perf_stats.csv").exists(),
            'hpa_configs': (self.project_root / "infra" / "k8s-manifests" / "hpa").exists()
        }
        self.validation_results['sprint_8'] = perf_results
        
        # Sprint 9: Testing & Quality Assurance
        print("🧪 Validating Sprint 9: Testing")
        testing_results = {
            'cypress_tests': len(list((self.project_root / "frontend" / "cypress" / "e2e").glob("*.cy.ts"))) > 0,
            'backend_tests': len(list((self.project_root / "backend" / "test").glob("*.e2e-spec.ts"))) > 0,
            'locust_tests': (self.project_root / "tests" / "perf" / "locustfile.py").exists()
        }
        self.validation_results['sprint_9'] = testing_results
        
        # Sprint 10: Documentation
        print("📚 Validating Sprint 10: Documentation")
        doc_results = {
            'complete_guide': (self.project_root / "CLOUDFORGE_AI_COMPLETE_GUIDE.md").exists(),
            'readme_updated': (self.project_root / "README.md").stat().st_size > 1000 if (self.project_root / "README.md").exists() else False,
            'api_docs': any(self.project_root.rglob("*api*.md"))
        }
        self.validation_results['sprint_10'] = doc_results
        
        # Sprint 11: Deployment & DevOps
        print("🚀 Validating Sprint 11: Deployment")
        deployment_results = {
            'kong_config': (self.project_root / "infra" / "kong" / "kong.yml").exists(),
            'ingress_config': (self.project_root / "helm-chart" / "templates" / "ingress.yaml").exists(),
            'worker_jobs': (self.project_root / "helm-chart" / "templates" / "worker-job.yaml").exists()
        }
        self.validation_results['sprint_11'] = deployment_results
        
        # Sprint 12: Final Integration
        print("🎯 Validating Sprint 12: Final Integration")
        integration_results = {
            'all_services_dockerized': all([
                (self.project_root / "frontend" / "Dockerfile").exists(),
                (self.project_root / "backend" / "Dockerfile").exists(),
                (self.project_root / "ai-scripts" / "Dockerfile").exists()
            ]),
            'demo_scripts': (self.project_root / "scripts" / "demo.ps1").exists(),
            'validation_scripts': len(list((self.project_root / "scripts").glob("validate_*.py"))) > 0
        }
        self.validation_results['sprint_12'] = integration_results
        
    def count_docker_services(self, compose_path):
        """Count services in docker-compose.yml"""
        try:
            import yaml
            with open(compose_path, 'r') as f:
                compose_data = yaml.safe_load(f)
                return len(compose_data.get('services', {}))
        except:
            # Fallback: count service definitions
            with open(compose_path, 'r') as f:
                content = f.read()
                return content.count('container_name:')
    
    def count_requirements(self, req_path):
        """Count packages in requirements.txt"""
        try:
            with open(req_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                return len(lines)
        except:
            return 0
    
    def generate_sprint_report(self):
        """Generate comprehensive sprint validation report"""
        print("\n" + "="*60)
        print("CLOUDFORGE AI SPRINT VALIDATION REPORT")
        print("="*60)
        
        # Run all sprint validations
        self.validate_sprint_1()
        self.validate_sprint_2()
        self.validate_sprint_3()
        self.validate_sprint_4()
        self.validate_sprint_5()
        self.validate_sprint_6()
        self.validate_sprint_7_to_12()
        
        # Calculate overall progress
        total_checks = 0
        passed_checks = 0
        
        for sprint, results in self.validation_results.items():
            for category, checks in results.items():
                if isinstance(checks, dict):
                    for check, status in checks.items():
                        total_checks += 1
                        if status:
                            passed_checks += 1
                elif isinstance(checks, bool):
                    total_checks += 1
                    if checks:
                        passed_checks += 1
                elif isinstance(checks, (int, list)):
                    total_checks += 1
                    if checks:
                        passed_checks += 1
        
        completion_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n📊 OVERALL PROGRESS")
        print(f"Checks passed: {passed_checks}/{total_checks}")
        print(f"Completion: {completion_percentage:.1f}%")
        
        # Save detailed report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_stats': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'completion_percentage': completion_percentage
            },
            'sprint_results': self.validation_results
        }
        
        with open('sprint_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: sprint_validation_report.json")
        
        return report

def main():
    """Main validation entry point"""
    project_root = os.getcwd()
    validator = SprintValidator(project_root)
    
    print("🚀 Starting CloudForge AI Sprint Validation")
    print(f"📁 Project root: {project_root}")
    
    report = validator.generate_sprint_report()
    
    # Print sprint summary
    print(f"\n📋 SPRINT SUMMARY")
    for sprint, results in validator.validation_results.items():
        sprint_num = sprint.replace('sprint_', '').upper()
        print(f"  {sprint_num}: {'✅' if any(results.values()) else '❌'}")
    
    return report

if __name__ == "__main__":
    main()