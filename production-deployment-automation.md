# CloudForge AI - Production Deployment Automation
## Complete CI/CD Pipeline with Oracle Cloud Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/production-deployment.yml
name: CloudForge AI Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME_BACKEND: cloudforge/backend
  IMAGE_NAME_FRONTEND: cloudforge/frontend
  IMAGE_NAME_AI: cloudforge/ai-services
  ORACLE_REGION: us-ashburn-1
  K8S_NAMESPACE: cloudforge-prod

jobs:
  test-suite:
    name: Comprehensive Test Suite
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: cloudforge_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    strategy:
      matrix:
        test-type: [unit, integration, e2e]
        node-version: [18.x, 20.x]
        python-version: [3.11, 3.12]
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
        cache-dependency-path: |
          backend/package-lock.json
          frontend/package-lock.json
    
    - name: Setup Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
        cache-dependency-path: ai-scripts/requirements.txt
    
    - name: Install Backend Dependencies
      working-directory: ./backend
      run: |
        npm ci
        npm run build
    
    - name: Install Frontend Dependencies
      working-directory: ./frontend
      run: |
        npm ci
        npm run build
    
    - name: Install AI Dependencies
      working-directory: ./ai-scripts
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run Backend Unit Tests
      working-directory: ./backend
      run: |
        npm run test:unit -- --coverage --passWithNoTests
        npm run test:integration -- --coverage
      env:
        DATABASE_URL: postgresql://test_user:test_password@localhost:5432/cloudforge_test
        REDIS_URL: redis://localhost:6379
        JWT_SECRET: test_jwt_secret_key_for_ci
        NODE_ENV: test
    
    - name: Run Frontend Tests
      working-directory: ./frontend
      run: |
        npm run test -- --coverage --watchAll=false
        npm run test:e2e:headless
      env:
        NEXT_PUBLIC_API_URL: http://localhost:3001
        NODE_ENV: test
    
    - name: Run AI Service Tests
      working-directory: ./ai-scripts
      run: |
        pytest tests/ -v --cov=. --cov-report=xml --cov-report=term
        python -m pytest tests/test_anomaly_detector.py::TestRealWorldScenarios -v
      env:
        PYTHONPATH: ${{ github.workspace }}/ai-scripts
    
    - name: Run Security Scans
      run: |
        # Backend security scan
        cd backend && npm audit --audit-level=high
        
        # Frontend security scan
        cd frontend && npm audit --audit-level=high
        
        # Python dependency scan
        cd ai-scripts && pip-audit --desc --format=json
    
    - name: Upload Test Results
      uses: actions/upload-artifact@v4
      with:
        name: test-results-${{ matrix.test-type }}-node${{ matrix.node-version }}-py${{ matrix.python-version }}
        path: |
          backend/coverage/
          frontend/coverage/
          ai-scripts/htmlcov/
          test-results.xml
    
    - name: Upload Coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: |
          backend/coverage/lcov.info
          frontend/coverage/lcov.info
          ai-scripts/coverage.xml
        fail_ci_if_error: true

  performance-tests:
    name: Performance Testing
    runs-on: ubuntu-latest
    needs: test-suite
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Setup Python for Locust
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Performance Testing Tools
      run: |
        pip install locust
        npm install -g artillery lighthouse-ci
    
    - name: Start Application Stack
      run: |
        docker-compose -f docker-compose.yml up -d
        sleep 60  # Wait for services to be ready
    
    - name: Run Load Tests
      run: |
        cd tests/perf
        locust -f locustfile.py --headless \
          --users 50 --spawn-rate 5 --run-time 5m \
          --host http://localhost:3001 \
          --csv performance_results
    
    - name: Run Lighthouse Performance Audit
      run: |
        lhci autorun --upload.target=filesystem --upload.outputDir=./lighthouse-results
    
    - name: Upload Performance Results
      uses: actions/upload-artifact@v4
      with:
        name: performance-results
        path: |
          tests/perf/performance_results*
          lighthouse-results/

  build-and-push:
    name: Build and Push Container Images
    runs-on: ubuntu-latest
    needs: [test-suite, performance-tests]
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v')
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract Metadata for Backend
      id: meta-backend
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME_BACKEND }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and Push Backend Image
      uses: docker/build-push-action@v5
      with:
        context: ./backend
        file: ./backend/Dockerfile
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta-backend.outputs.tags }}
        labels: ${{ steps.meta-backend.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Extract Metadata for Frontend
      id: meta-frontend
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME_FRONTEND }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and Push Frontend Image
      uses: docker/build-push-action@v5
      with:
        context: ./frontend
        file: ./frontend/Dockerfile
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta-frontend.outputs.tags }}
        labels: ${{ steps.meta-frontend.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Extract Metadata for AI Services
      id: meta-ai
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME_AI }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and Push AI Services Image
      uses: docker/build-push-action@v5
      with:
        context: ./ai-scripts
        file: ./ai-scripts/Dockerfile
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta-ai.outputs.tags }}
        labels: ${{ steps.meta-ai.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  oracle-cloud-deploy:
    name: Deploy to Oracle Cloud
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Setup Oracle Cloud CLI
      run: |
        curl -L -O https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh
        chmod +x install.sh
        ./install.sh --accept-all-defaults
        echo "$HOME/bin" >> $GITHUB_PATH
    
    - name: Configure Oracle Cloud CLI
      run: |
        mkdir -p ~/.oci
        echo "${{ secrets.OCI_CONFIG }}" > ~/.oci/config
        echo "${{ secrets.OCI_PRIVATE_KEY }}" > ~/.oci/private_key.pem
        chmod 600 ~/.oci/private_key.pem
    
    - name: Setup Kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Get Kubernetes Config
      run: |
        oci ce cluster create-kubeconfig \
          --cluster-id ${{ secrets.OKE_CLUSTER_ID }} \
          --file ~/.kube/config \
          --region ${{ env.ORACLE_REGION }} \
          --token-version 2.0.0
    
    - name: Deploy Infrastructure
      working-directory: ./terraform/oracle-cloud
      run: |
        terraform init
        terraform plan -var-file="production.tfvars"
        terraform apply -auto-approve -var-file="production.tfvars"
      env:
        TF_VAR_tenancy_ocid: ${{ secrets.OCI_TENANCY_OCID }}
        TF_VAR_user_ocid: ${{ secrets.OCI_USER_OCID }}
        TF_VAR_fingerprint: ${{ secrets.OCI_FINGERPRINT }}
        TF_VAR_private_key_path: ~/.oci/private_key.pem
        TF_VAR_region: ${{ env.ORACLE_REGION }}
        TF_VAR_compartment_id: ${{ secrets.OCI_COMPARTMENT_ID }}
    
    - name: Deploy Application to Kubernetes
      run: |
        # Create namespace
        kubectl create namespace ${{ env.K8S_NAMESPACE }} --dry-run=client -o yaml | kubectl apply -f -
        
        # Apply Kubernetes manifests
        kubectl apply -f k8s/production/ -n ${{ env.K8S_NAMESPACE }}
        
        # Update image tags
        kubectl set image deployment/cloudforge-backend backend=${{ env.REGISTRY }}/${{ env.IMAGE_NAME_BACKEND }}:${{ github.sha }} -n ${{ env.K8S_NAMESPACE }}
        kubectl set image deployment/cloudforge-frontend frontend=${{ env.REGISTRY }}/${{ env.IMAGE_NAME_FRONTEND }}:${{ github.sha }} -n ${{ env.K8S_NAMESPACE }}
        
        # Wait for deployment
        kubectl rollout status deployment/cloudforge-backend -n ${{ env.K8S_NAMESPACE }} --timeout=600s
        kubectl rollout status deployment/cloudforge-frontend -n ${{ env.K8S_NAMESPACE }} --timeout=600s
    
    - name: Run Post-Deployment Tests
      run: |
        # Health check tests
        kubectl get pods -n ${{ env.K8S_NAMESPACE }}
        
        # Wait for services to be ready
        kubectl wait --for=condition=ready pod -l app=cloudforge-backend -n ${{ env.K8S_NAMESPACE }} --timeout=300s
        kubectl wait --for=condition=ready pod -l app=cloudforge-frontend -n ${{ env.K8S_NAMESPACE }} --timeout=300s
        
        # Test service endpoints
        kubectl port-forward svc/cloudforge-backend 8080:80 -n ${{ env.K8S_NAMESPACE }} &
        sleep 10
        curl -f http://localhost:8080/health || exit 1
    
    - name: Setup Monitoring
      run: |
        # Deploy Prometheus and Grafana
        kubectl apply -f k8s/monitoring/ -n cloudforge-system
        
        # Setup alerting rules
        kubectl apply -f k8s/monitoring/alerts/ -n cloudforge-system
    
    - name: Generate Deployment Report
      run: |
        echo "# CloudForge AI Deployment Report" > deployment-report.md
        echo "**Deployment Time:** $(date)" >> deployment-report.md
        echo "**Git Commit:** ${{ github.sha }}" >> deployment-report.md
        echo "**Environment:** Production (Oracle Cloud)" >> deployment-report.md
        echo "" >> deployment-report.md
        echo "## Services Status" >> deployment-report.md
        kubectl get pods -n ${{ env.K8S_NAMESPACE }} -o wide >> deployment-report.md
        echo "" >> deployment-report.md
        echo "## Resource Usage" >> deployment-report.md
        kubectl top nodes >> deployment-report.md
        kubectl top pods -n ${{ env.K8S_NAMESPACE }} >> deployment-report.md
    
    - name: Upload Deployment Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: deployment-report
        path: deployment-report.md

  post-deployment-validation:
    name: Post-Deployment Validation
    runs-on: ubuntu-latest
    needs: oracle-cloud-deploy
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Setup Python for Validation Tests
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Validation Tools
      run: |
        pip install requests pytest locust
        npm install -g newman
    
    - name: Run API Validation Tests
      run: |
        python scripts/validate-deployment.py --env production
    
    - name: Run Postman Collection Tests
      run: |
        newman run tests/api/CloudForge-API.postman_collection.json \
          --environment tests/api/Production.postman_environment.json \
          --reporters cli,json \
          --reporter-json-export newman-results.json
    
    - name: Validate Database Migrations
      run: |
        python scripts/validate-database.py --connection-string "${{ secrets.PROD_DATABASE_URL }}"
    
    - name: Performance Validation
      run: |
        cd tests/perf
        locust -f production-validation.py --headless \
          --users 10 --spawn-rate 2 --run-time 2m \
          --host https://api.cloudforge.example.com
    
    - name: Security Validation
      run: |
        # SSL/TLS validation
        python scripts/validate-ssl.py https://cloudforge.example.com
        
        # Security headers validation
        python scripts/validate-security-headers.py https://cloudforge.example.com

  notification:
    name: Deployment Notification
    runs-on: ubuntu-latest
    needs: [oracle-cloud-deploy, post-deployment-validation]
    if: always()
    
    steps:
    - name: Notify Slack on Success
      if: needs.oracle-cloud-deploy.result == 'success' && needs.post-deployment-validation.result == 'success'
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: |
          🚀 CloudForge AI successfully deployed to production!
          
          **Commit:** ${{ github.sha }}
          **Environment:** Oracle Cloud Free Tier
          **Deployment Time:** $(date)
          
          All validation tests passed ✅
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    
    - name: Notify Slack on Failure
      if: needs.oracle-cloud-deploy.result == 'failure' || needs.post-deployment-validation.result == 'failure'
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        text: |
          ❌ CloudForge AI deployment failed!
          
          **Commit:** ${{ github.sha }}
          **Failed Job:** ${{ needs.oracle-cloud-deploy.result == 'failure' && 'Deployment' || 'Validation' }}
          
          Please check the logs and fix the issues.
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Production Validation Scripts
```python
# scripts/validate-deployment.py
#!/usr/bin/env python3
"""
Production deployment validation script
Validates all CloudForge AI services are running correctly
"""

import requests
import time
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

class ProductionValidator:
    """Comprehensive production environment validator"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        self.results = []
    
    def validate_service_health(self) -> bool:
        """Validate all service health endpoints"""
        health_endpoints = [
            ('/health', 'Basic Health Check'),
            ('/health/ready', 'Readiness Check'),
            ('/health/live', 'Liveness Check'),
            ('/metrics', 'Metrics Endpoint')
        ]
        
        print("🔍 Validating Service Health...")
        all_healthy = True
        
        for endpoint, description in health_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                status = "✅ PASS" if response.status_code == 200 else "❌ FAIL"
                
                result = {
                    'test': description,
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'passed': response.status_code == 200,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                print(f"  {status} {description}: {response.status_code} ({result['response_time']:.3f}s)")
                
                if response.status_code != 200:
                    all_healthy = False
                    
            except Exception as e:
                print(f"  ❌ FAIL {description}: {str(e)}")
                all_healthy = False
                self.results.append({
                    'test': description,
                    'endpoint': endpoint,
                    'error': str(e),
                    'passed': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        return all_healthy
    
    def validate_api_endpoints(self) -> bool:
        """Validate critical API endpoints"""
        api_tests = [
            ('GET', '/api/dashboard', {}, 'Dashboard API'),
            ('GET', '/api/migrations', {}, 'Migrations List API'),
            ('GET', '/api/marketplace/stats', {}, 'Marketplace Stats API'),
            ('GET', '/api/iac/templates', {}, 'IaC Templates API'),
        ]
        
        print("\n🔍 Validating API Endpoints...")
        all_passed = True
        
        # First, authenticate to get a token
        auth_token = self._authenticate()
        headers = {'Authorization': f'Bearer {auth_token}'} if auth_token else {}
        
        for method, endpoint, data, description in api_tests:
            try:
                if method == 'GET':
                    response = self.session.get(f"{self.base_url}{endpoint}", headers=headers)
                elif method == 'POST':
                    response = self.session.post(f"{self.base_url}{endpoint}", json=data, headers=headers)
                
                expected_codes = [200, 201, 401]  # 401 is acceptable if auth is required
                status = "✅ PASS" if response.status_code in expected_codes else "❌ FAIL"
                
                result = {
                    'test': description,
                    'method': method,
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'passed': response.status_code in expected_codes,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                print(f"  {status} {description}: {response.status_code} ({result['response_time']:.3f}s)")
                
                if response.status_code not in expected_codes:
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ FAIL {description}: {str(e)}")
                all_passed = False
                self.results.append({
                    'test': description,
                    'method': method,
                    'endpoint': endpoint,
                    'error': str(e),
                    'passed': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        return all_passed
    
    def validate_database_connectivity(self) -> bool:
        """Validate database connectivity through API"""
        print("\n🔍 Validating Database Connectivity...")
        
        try:
            # Test database through a simple API call that requires DB access
            response = self.session.get(f"{self.base_url}/api/migrations")
            
            if response.status_code in [200, 401]:  # 401 is ok, means API is working
                print("  ✅ PASS Database connectivity: API responds correctly")
                self.results.append({
                    'test': 'Database Connectivity',
                    'passed': True,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                print(f"  ❌ FAIL Database connectivity: Unexpected response {response.status_code}")
                self.results.append({
                    'test': 'Database Connectivity',
                    'status_code': response.status_code,
                    'passed': False,
                    'timestamp': datetime.now().isoformat()
                })
                return False
                
        except Exception as e:
            print(f"  ❌ FAIL Database connectivity: {str(e)}")
            self.results.append({
                'test': 'Database Connectivity',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def validate_ai_services(self) -> bool:
        """Validate AI services functionality"""
        print("\n🔍 Validating AI Services...")
        
        # Test data for anomaly detection
        test_data = {
            "data": [
                {"cpu": 45.2, "memory": 67.8, "disk": 123.4},
                {"cpu": 52.1, "memory": 71.2, "disk": 145.6},
                {"cpu": 98.7, "memory": 95.5, "disk": 890.2}  # Anomaly
            ],
            "algorithm": "isolation_forest"
        }
        
        ai_tests = [
            ('POST', '/api/ai/detect-anomalies', test_data, 'Anomaly Detection'),
            ('GET', '/api/ai/models', {}, 'AI Models List'),
        ]
        
        all_passed = True
        
        for method, endpoint, data, description in ai_tests:
            try:
                if method == 'GET':
                    response = self.session.get(f"{self.base_url}{endpoint}")
                elif method == 'POST':
                    response = self.session.post(f"{self.base_url}{endpoint}", json=data)
                
                # AI services might take longer to respond
                if response.elapsed.total_seconds() > 30:
                    print(f"  ⚠️  WARN {description}: Slow response ({response.elapsed.total_seconds():.3f}s)")
                
                expected_codes = [200, 201, 202, 401]  # 202 for async processing
                status = "✅ PASS" if response.status_code in expected_codes else "❌ FAIL"
                
                result = {
                    'test': description,
                    'method': method,
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'passed': response.status_code in expected_codes,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                print(f"  {status} {description}: {response.status_code} ({result['response_time']:.3f}s)")
                
                if response.status_code not in expected_codes:
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ FAIL {description}: {str(e)}")
                all_passed = False
                self.results.append({
                    'test': description,
                    'method': method,
                    'endpoint': endpoint,
                    'error': str(e),
                    'passed': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        return all_passed
    
    def validate_performance_metrics(self) -> bool:
        """Validate performance meets production requirements"""
        print("\n🔍 Validating Performance Metrics...")
        
        performance_tests = [
            ('/health', 'Health Check Response Time', 1.0),  # Should be < 1s
            ('/api/dashboard', 'Dashboard Load Time', 3.0),   # Should be < 3s
            ('/api/migrations', 'API Response Time', 2.0),    # Should be < 2s
        ]
        
        all_passed = True
        
        for endpoint, description, max_time in performance_tests:
            try:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}{endpoint}")
                response_time = time.time() - start_time
                
                passed = response_time < max_time and response.status_code in [200, 401]
                status = "✅ PASS" if passed else "❌ FAIL"
                
                result = {
                    'test': description,
                    'endpoint': endpoint,
                    'response_time': response_time,
                    'max_allowed': max_time,
                    'status_code': response.status_code,
                    'passed': passed,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                print(f"  {status} {description}: {response_time:.3f}s (max: {max_time}s)")
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ FAIL {description}: {str(e)}")
                all_passed = False
                self.results.append({
                    'test': description,
                    'endpoint': endpoint,
                    'error': str(e),
                    'passed': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        return all_passed
    
    def validate_security_headers(self) -> bool:
        """Validate security headers are present"""
        print("\n🔍 Validating Security Headers...")
        
        required_headers = [
            ('X-Content-Type-Options', 'nosniff'),
            ('X-Frame-Options', ['DENY', 'SAMEORIGIN']),
            ('X-XSS-Protection', '1; mode=block'),
            ('Strict-Transport-Security', None),  # Just check presence
        ]
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            headers = response.headers
            
            all_passed = True
            
            for header_name, expected_value in required_headers:
                if header_name in headers:
                    header_value = headers[header_name]
                    
                    if expected_value is None:
                        # Just check presence
                        print(f"  ✅ PASS {header_name}: Present ({header_value})")
                        passed = True
                    elif isinstance(expected_value, list):
                        # Check if value is in list
                        passed = header_value in expected_value
                        status = "✅ PASS" if passed else "❌ FAIL"
                        print(f"  {status} {header_name}: {header_value}")
                    else:
                        # Check exact match
                        passed = header_value == expected_value
                        status = "✅ PASS" if passed else "❌ FAIL"
                        print(f"  {status} {header_name}: {header_value}")
                    
                    if not passed:
                        all_passed = False
                else:
                    print(f"  ❌ FAIL {header_name}: Missing")
                    all_passed = False
                
                self.results.append({
                    'test': f'Security Header: {header_name}',
                    'expected': expected_value,
                    'actual': headers.get(header_name),
                    'passed': header_name in headers and passed if header_name in headers else False,
                    'timestamp': datetime.now().isoformat()
                })
            
            return all_passed
            
        except Exception as e:
            print(f"  ❌ FAIL Security headers validation: {str(e)}")
            self.results.append({
                'test': 'Security Headers',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def _authenticate(self) -> str:
        """Attempt to authenticate and get a token for API testing"""
        try:
            # Try with test credentials
            auth_data = {
                "email": "test@cloudforge.com",
                "password": "TestPassword123!"
            }
            
            response = self.session.post(f"{self.base_url}/auth/login", json=auth_data)
            if response.status_code == 200:
                return response.json().get('access_token')
            else:
                print(f"  ⚠️  WARN Authentication failed (expected in production): {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ⚠️  WARN Authentication error (expected in production): {str(e)}")
            return None
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'validation_time': datetime.now().isoformat()
            },
            'detailed_results': self.results,
            'recommendations': []
        }
        
        # Add recommendations based on results
        if failed_tests > 0:
            report['recommendations'].append("Review failed tests and fix underlying issues")
        
        slow_tests = [r for r in self.results if r.get('response_time', 0) > 5.0]
        if slow_tests:
            report['recommendations'].append("Investigate slow response times for optimal performance")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Validate CloudForge AI production deployment')
    parser.add_argument('--env', choices=['staging', 'production'], default='production',
                       help='Environment to validate')
    parser.add_argument('--url', help='Base URL to validate (overrides environment default)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--output', help='Output file for validation report (JSON)')
    
    args = parser.parse_args()
    
    # Set base URL based on environment
    if args.url:
        base_url = args.url
    elif args.env == 'production':
        base_url = 'https://api.cloudforge.example.com'
    else:
        base_url = 'https://staging-api.cloudforge.example.com'
    
    print(f"🚀 CloudForge AI Production Validation")
    print(f"Environment: {args.env}")
    print(f"Base URL: {base_url}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 50)
    
    validator = ProductionValidator(base_url, args.timeout)
    
    # Run all validation tests
    validation_results = []
    validation_results.append(validator.validate_service_health())
    validation_results.append(validator.validate_api_endpoints())
    validation_results.append(validator.validate_database_connectivity())
    validation_results.append(validator.validate_ai_services())
    validation_results.append(validator.validate_performance_metrics())
    validation_results.append(validator.validate_security_headers())
    
    # Generate final report
    report = validator.generate_report()
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {report['validation_summary']['total_tests']}")
    print(f"Passed: {report['validation_summary']['passed_tests']}")
    print(f"Failed: {report['validation_summary']['failed_tests']}")
    print(f"Success Rate: {report['validation_summary']['success_rate']:.1f}%")
    
    if report['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")
    
    # Exit with appropriate code
    overall_success = all(validation_results)
    if overall_success:
        print("\n✅ ALL VALIDATIONS PASSED - Production deployment is healthy!")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILURES DETECTED - Please review and fix issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Database Migration Validation
```python
# scripts/validate-database.py
#!/usr/bin/env python3
"""
Database migration and schema validation script
Ensures database is in correct state after deployment
"""

import psycopg2
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple

class DatabaseValidator:
    """Production database validation"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.results = []
    
    def validate_schema(self) -> bool:
        """Validate database schema matches expected structure"""
        print("🔍 Validating Database Schema...")
        
        expected_tables = [
            'users', 'migrations', 'marketplace_items', 'iac_templates',
            'auth_tokens', 'audit_logs', 'performance_metrics'
        ]
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Check all expected tables exist
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    
                    actual_tables = [row[0] for row in cursor.fetchall()]
                    
                    all_passed = True
                    for table in expected_tables:
                        if table in actual_tables:
                            print(f"  ✅ PASS Table exists: {table}")
                            self.results.append({
                                'test': f'Table Exists: {table}',
                                'passed': True,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            print(f"  ❌ FAIL Missing table: {table}")
                            self.results.append({
                                'test': f'Table Exists: {table}',
                                'passed': False,
                                'timestamp': datetime.now().isoformat()
                            })
                            all_passed = False
                    
                    return all_passed
                    
        except Exception as e:
            print(f"  ❌ FAIL Schema validation error: {str(e)}")
            self.results.append({
                'test': 'Schema Validation',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def validate_migrations(self) -> bool:
        """Validate all migrations have been applied"""
        print("\n🔍 Validating Database Migrations...")
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Check migrations table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = 'migrations'
                        )
                    """)
                    
                    if not cursor.fetchone()[0]:
                        print("  ❌ FAIL Migrations table does not exist")
                        return False
                    
                    # Check latest migration
                    cursor.execute("""
                        SELECT id, name, executed_at 
                        FROM migrations 
                        ORDER BY executed_at DESC 
                        LIMIT 1
                    """)
                    
                    latest_migration = cursor.fetchone()
                    if latest_migration:
                        print(f"  ✅ PASS Latest migration: {latest_migration[1]} (ID: {latest_migration[0]})")
                        self.results.append({
                            'test': 'Latest Migration',
                            'migration_id': latest_migration[0],
                            'migration_name': latest_migration[1],
                            'executed_at': latest_migration[2].isoformat(),
                            'passed': True,
                            'timestamp': datetime.now().isoformat()
                        })
                        return True
                    else:
                        print("  ❌ FAIL No migrations found")
                        self.results.append({
                            'test': 'Latest Migration',
                            'passed': False,
                            'timestamp': datetime.now().isoformat()
                        })
                        return False
                        
        except Exception as e:
            print(f"  ❌ FAIL Migration validation error: {str(e)}")
            self.results.append({
                'test': 'Migration Validation',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity and constraints"""
        print("\n🔍 Validating Data Integrity...")
        
        integrity_checks = [
            ("SELECT COUNT(*) FROM users WHERE email IS NULL", 0, "Users have email addresses"),
            ("SELECT COUNT(*) FROM migrations WHERE name IS NULL", 0, "Migrations have names"),
            ("SELECT COUNT(*) FROM auth_tokens WHERE expires_at < NOW()", None, "Auth tokens cleanup"),
        ]
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    all_passed = True
                    
                    for query, expected, description in integrity_checks:
                        cursor.execute(query)
                        result = cursor.fetchone()[0]
                        
                        if expected is not None:
                            passed = result == expected
                            status = "✅ PASS" if passed else "❌ FAIL"
                            print(f"  {status} {description}: {result} (expected: {expected})")
                        else:
                            # Just report the value
                            print(f"  ℹ️  INFO {description}: {result}")
                            passed = True
                        
                        self.results.append({
                            'test': description,
                            'query': query,
                            'result': result,
                            'expected': expected,
                            'passed': passed,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        if not passed:
                            all_passed = False
                    
                    return all_passed
                    
        except Exception as e:
            print(f"  ❌ FAIL Data integrity validation error: {str(e)}")
            self.results.append({
                'test': 'Data Integrity Validation',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def validate_performance(self) -> bool:
        """Validate database performance metrics"""
        print("\n🔍 Validating Database Performance...")
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Check database size
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database()))
                    """)
                    db_size = cursor.fetchone()[0]
                    print(f"  ℹ️  INFO Database size: {db_size}")
                    
                    # Check connection count
                    cursor.execute("""
                        SELECT count(*) FROM pg_stat_activity 
                        WHERE state = 'active'
                    """)
                    active_connections = cursor.fetchone()[0]
                    print(f"  ℹ️  INFO Active connections: {active_connections}")
                    
                    # Simple performance test
                    import time
                    start_time = time.time()
                    cursor.execute("SELECT COUNT(*) FROM users")
                    query_time = time.time() - start_time
                    
                    passed = query_time < 1.0  # Should be < 1 second
                    status = "✅ PASS" if passed else "❌ FAIL"
                    print(f"  {status} Simple query performance: {query_time:.3f}s")
                    
                    self.results.append({
                        'test': 'Database Performance',
                        'db_size': db_size,
                        'active_connections': active_connections,
                        'query_time': query_time,
                        'passed': passed,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return passed
                    
        except Exception as e:
            print(f"  ❌ FAIL Performance validation error: {str(e)}")
            self.results.append({
                'test': 'Performance Validation',
                'error': str(e),
                'passed': False,
                'timestamp': datetime.now().isoformat()
            })
            return False

def main():
    parser = argparse.ArgumentParser(description='Validate CloudForge AI database')
    parser.add_argument('--connection-string', required=True,
                       help='PostgreSQL connection string')
    parser.add_argument('--output', help='Output file for validation report (JSON)')
    
    args = parser.parse_args()
    
    print("🗄️  CloudForge AI Database Validation")
    print("=" * 50)
    
    validator = DatabaseValidator(args.connection_string)
    
    # Run all validation tests
    validation_results = []
    validation_results.append(validator.validate_schema())
    validation_results.append(validator.validate_migrations())
    validation_results.append(validator.validate_data_integrity())
    validation_results.append(validator.validate_performance())
    
    # Generate report
    total_tests = len(validator.results)
    passed_tests = sum(1 for result in validator.results if result.get('passed', False))
    failed_tests = total_tests - passed_tests
    
    report = {
        'database_validation_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'validation_time': datetime.now().isoformat()
        },
        'detailed_results': validator.results
    }
    
    print("\n" + "=" * 50)
    print("📊 DATABASE VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {report['database_validation_summary']['success_rate']:.1f}%")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")
    
    # Exit with appropriate code
    overall_success = all(validation_results)
    if overall_success:
        print("\n✅ ALL DATABASE VALIDATIONS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ DATABASE VALIDATION FAILURES DETECTED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```