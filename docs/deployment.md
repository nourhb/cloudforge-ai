# CloudForge AI - Quick Deployment Guide
# Production-Ready Deployment for University & Enterprise Environments
# Version: 2.0.0
# Date: October 1, 2025

## 🚀 Quick Start Deployment Guide

This guide provides step-by-step instructions for deploying CloudForge AI in various environments, from local development to production Oracle Cloud deployment.

### 📋 Prerequisites

Before deploying CloudForge AI, ensure you have the following tools and resources:

#### System Requirements

**Minimum Local Development:**
- 8GB RAM
- 4 CPU cores
- 50GB available storage
- Windows 10/11, macOS 10.15+, or Ubuntu 18.04+

**Production Oracle Cloud Free Tier:**
- 2x ARM-based Ampere A1 compute instances
- 4 OCPU total (Always Free eligible)
- 24GB RAM total
- 200GB Block Volume storage

#### Required Tools

```powershell
# Install required tools (Windows PowerShell)

# 1. Docker Desktop
winget install Docker.DockerDesktop

# 2. Kubernetes CLI
winget install Kubernetes.kubectl

# 3. Helm Package Manager
winget install Helm.Helm

# 4. Git
winget install Git.Git

# 5. Node.js & npm
winget install OpenJS.NodeJS

# 6. Python 3.9+
winget install Python.Python.3.11

# 7. VS Code (optional but recommended)
winget install Microsoft.VisualStudioCode
```

#### Cloud Provider Setup (Optional)

**Oracle Cloud Free Tier Account:**
1. Sign up at https://www.oracle.com/cloud/free/
2. Create Always Free eligible compute instances
3. Configure VCN and security lists
4. Generate SSH key pairs for access

### 🏠 Local Development Deployment

#### Option 1: Docker Compose (Recommended for Quick Start)

```powershell
# Clone the repository
git clone https://github.com/your-org/cloudforge-ai.git
cd cloudforge-ai

# Set up environment variables
cp .env.example .env
# Edit .env file with your configuration

# Start all services
docker-compose up -d

# Verify deployment
docker-compose ps
```

**Expected Output:**
```
NAME                        COMMAND                  STATUS
cloudforge-ai-frontend-1    "docker-entrypoint.s…"  Up 30 seconds
cloudforge-ai-backend-1     "docker-entrypoint.s…"  Up 30 seconds
cloudforge-ai-ai-services-1 "python app.py"         Up 30 seconds
cloudforge-ai-postgres-1    "docker-entrypoint.s…"  Up 45 seconds
cloudforge-ai-redis-1       "docker-entrypoint.s…"  Up 45 seconds
cloudforge-ai-minio-1       "sh -c 'mkdir -p /dat…"  Up 45 seconds
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- AI Services: http://localhost:5000
- MinIO Console: http://localhost:9001
- Admin credentials: minioadmin / minioadmin

#### Option 2: Kubernetes with Minikube

```powershell
# Start Minikube
minikube start --memory=8192 --cpus=4 --disk-size=20g

# Enable necessary addons
minikube addons enable ingress
minikube addons enable metrics-server

# Deploy using Helm
helm repo add cloudforge-ai ./helm-chart
helm install cloudforge-ai ./helm-chart --values helm-chart/values.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod --all --timeout=300s

# Access the application
minikube service cloudforge-ai-frontend --url
```

### ☁️ Oracle Cloud Production Deployment

#### Automated Deployment with Ansible

```powershell
# Navigate to infrastructure directory
cd infra

# Install Ansible (if not already installed)
pip install ansible

# Configure Oracle Cloud credentials
# Edit infra/ansible/group_vars/all.yml with your Oracle Cloud details

# Run the automated deployment
ansible-playbook -i ansible/inventory/oracle.yml ansible/deploy_oracle.yml

# Monitor deployment progress
ansible-playbook -i ansible/inventory/oracle.yml ansible/deploy_oracle.yml --tags "status"
```

#### Manual Oracle Cloud Deployment

**Step 1: Create Compute Instances**

```bash
# Create ARM-based compute instances
oci compute instance launch \
  --compartment-id <your-compartment-id> \
  --availability-domain <availability-domain> \
  --shape VM.Standard.A1.Flex \
  --shape-config '{"ocpus": 2, "memory_in_gbs": 12}' \
  --image-id <ubuntu-22.04-arm64-image-id> \
  --subnet-id <your-subnet-id> \
  --assign-public-ip true \
  --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
  --display-name cloudforge-ai-node-1

# Repeat for second instance (cloudforge-ai-node-2)
```

**Step 2: Install Kubernetes (K3s)**

```bash
# On first node (master)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -

# Get node token
sudo cat /var/lib/rancher/k3s/server/node-token

# On second node (worker)
curl -sfL https://get.k3s.io | K3S_URL=https://<master-ip>:6443 K3S_TOKEN=<node-token> sh -
```

**Step 3: Deploy CloudForge AI**

```bash
# Copy kubeconfig
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-arm64.tar.gz | tar -xzO linux-arm64/helm > /tmp/helm
sudo mv /tmp/helm /usr/local/bin/helm
sudo chmod +x /usr/local/bin/helm

# Deploy application
git clone https://github.com/your-org/cloudforge-ai.git
cd cloudforge-ai
helm install cloudforge-ai ./helm-chart --values helm-chart/values-oracle.yaml

# Configure ingress
kubectl apply -f infra/k8s-manifests/ingress.yaml
```

### 🔧 Configuration

#### Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/cloudforge_ai
REDIS_URL=redis://redis:6379

# AI Services Configuration
HUGGINGFACE_API_TOKEN=your_huggingface_token
AI_SERVICE_URL=http://ai-services:5000

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_ENDPOINT=minio:9000

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRES_IN=24h

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Monitoring Configuration
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Oracle Cloud Configuration (for production)
OCI_TENANCY_ID=your_tenancy_ocid
OCI_USER_ID=your_user_ocid
OCI_FINGERPRINT=your_key_fingerprint
OCI_PRIVATE_KEY_FILE=/path/to/private/key
OCI_REGION=us-ashburn-1
```

#### Helm Values Configuration

**Local Development (`values-local.yaml`):**

```yaml
# Resource allocation for local development
resources:
  frontend:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  backend:
    requests:
      memory: "512Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  ai-services:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: localhost
      paths:
        - path: /
          pathType: Prefix

# Storage configuration
storage:
  minio:
    persistence:
      enabled: true
      size: 10Gi
  postgresql:
    persistence:
      enabled: true
      size: 5Gi
```

**Oracle Cloud Production (`values-oracle.yaml`):**

```yaml
# Resource allocation for Oracle Cloud Free Tier
resources:
  frontend:
    requests:
      memory: "512Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  backend:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1500m"
  ai-services:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

# High availability configuration
replicaCount:
  frontend: 2
  backend: 3
  ai-services: 2

# Ingress with TLS
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
  tls:
    - secretName: cloudforge-ai-tls
      hosts:
        - cloudforge-ai.yourdomain.com

# Storage configuration
storage:
  minio:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: "oci-bv"
  postgresql:
    persistence:
      enabled: true
      size: 50Gi
      storageClass: "oci-bv"

# Monitoring
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

### 🔍 Verification and Testing

#### Health Checks

```powershell
# Check service health
curl http://localhost:8000/health
curl http://localhost:5000/health

# Check database connectivity
curl http://localhost:8000/api/health/db

# Check AI services
curl http://localhost:5000/api/health/models
```

#### Load Testing

```powershell
# Install Locust
pip install locust

# Run performance tests
cd tests/perf
locust -f locustfile.py --host=http://localhost:8000
```

#### Security Testing

```powershell
# Run OWASP ZAP security scan
cd tests/security
.\scan.zap.ps1 -target http://localhost:3000
```

### 📊 Monitoring Setup

#### Prometheus Metrics

Access Prometheus at http://localhost:9090 (local) or your configured domain.

**Key Metrics to Monitor:**
- `cloudforge_api_requests_total` - Total API requests
- `cloudforge_api_duration_seconds` - Request duration
- `cloudforge_db_connections_active` - Active database connections
- `cloudforge_ai_processing_seconds` - AI processing time

#### Grafana Dashboards

Access Grafana at http://localhost:3000 (local) with admin/admin credentials.

**Pre-configured Dashboards:**
1. **System Overview** - Overall system health and performance
2. **API Performance** - Request rates, response times, error rates
3. **Database Metrics** - Connection pools, query performance, storage usage
4. **AI Services** - Model performance, processing queues, accuracy metrics
5. **Infrastructure** - CPU, memory, disk, network utilization

### 🔧 Troubleshooting

#### Common Issues

**1. Database Connection Issues**

```powershell
# Check PostgreSQL status
kubectl get pods -l app=postgresql

# View database logs
kubectl logs -f deployment/postgresql

# Test database connection
kubectl exec -it deployment/backend -- npm run db:test
```

**2. AI Services Not Starting**

```powershell
# Check AI services logs
kubectl logs -f deployment/ai-services

# Verify model downloads
kubectl exec -it deployment/ai-services -- python -c "from transformers import pipeline; print('Models loaded successfully')"

# Check disk space for model storage
kubectl exec -it deployment/ai-services -- df -h
```

**3. MinIO Storage Issues**

```powershell
# Check MinIO status
kubectl get pods -l app=minio

# Access MinIO console
kubectl port-forward service/minio-console 9001:9001

# Test bucket creation
kubectl exec -it deployment/minio -- mc mb local/test-bucket
```

**4. Frontend Build Issues**

```powershell
# Check Node.js version
kubectl exec -it deployment/frontend -- node --version

# Rebuild frontend
kubectl exec -it deployment/frontend -- npm run build

# Check environment variables
kubectl exec -it deployment/frontend -- env | grep NEXT_PUBLIC
```

#### Log Analysis

```powershell
# View aggregated logs
kubectl logs -f -l app=cloudforge-ai --tail=100

# Filter error logs
kubectl logs -l app=cloudforge-ai | grep ERROR

# Export logs for analysis
kubectl logs -l app=cloudforge-ai --since=1h > deployment-logs.txt
```

### 🔄 Updates and Maintenance

#### Rolling Updates

```powershell
# Update application version
helm upgrade cloudforge-ai ./helm-chart --set image.tag=v2.1.0

# Check rollout status
kubectl rollout status deployment/backend
kubectl rollout status deployment/frontend
kubectl rollout status deployment/ai-services

# Rollback if needed
helm rollback cloudforge-ai 1
```

#### Database Migrations

```powershell
# Run database migrations
kubectl exec -it deployment/backend -- npm run migrate

# Backup database before migration
kubectl exec -it deployment/postgresql -- pg_dump -U postgres cloudforge_ai > backup.sql

# Verify migration status
kubectl exec -it deployment/backend -- npm run migrate:status
```

#### Backup and Recovery

```powershell
# Create full system backup
helm get values cloudforge-ai > config-backup.yaml
kubectl get pvc -o yaml > storage-backup.yaml

# Schedule automated backups
kubectl apply -f infra/k8s-manifests/backup-cronjob.yaml

# Test disaster recovery
kubectl delete namespace cloudforge-ai
helm install cloudforge-ai ./helm-chart --values config-backup.yaml
```

### 📈 Performance Optimization

#### Database Optimization

```sql
-- Optimize frequently used queries
CREATE INDEX CONCURRENTLY idx_projects_user_id ON projects(user_id);
CREATE INDEX CONCURRENTLY idx_migrations_status ON migrations(status);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp ON audit_logs(created_at);

-- Enable query performance monitoring
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
```

#### Redis Optimization

```redis
# Configure memory optimization
CONFIG SET maxmemory 2gb
CONFIG SET maxmemory-policy allkeys-lru

# Enable persistence
CONFIG SET save "900 1 300 10 60 10000"

# Monitor performance
INFO memory
INFO stats
```

#### AI Services Optimization

```python
# Optimize model loading in production
import torch
from transformers import pipeline, set_seed

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load models with optimization
migration_analyzer = pipeline(
    "text-generation",
    model="distilgpt2",
    device=device,
    torch_dtype=torch.float16,  # Use half precision
    return_full_text=False
)

# Enable model caching
migration_analyzer.save_pretrained("/app/models/distilgpt2-cached")
```

This deployment guide provides comprehensive instructions for getting CloudForge AI running in various environments, from local development to production Oracle Cloud deployment. The guide includes troubleshooting, monitoring, and optimization recommendations for a complete production-ready deployment experience.

### 🎯 Quick Reference Commands

**Start Local Development:**
```powershell
git clone https://github.com/your-org/cloudforge-ai.git
cd cloudforge-ai
docker-compose up -d
```

**Deploy to Oracle Cloud:**
```powershell
cd infra
ansible-playbook -i ansible/inventory/oracle.yml ansible/deploy_oracle.yml
```

**Check System Health:**
```powershell
curl http://localhost:8000/health
kubectl get pods
helm status cloudforge-ai
```

**View Logs:**
```powershell
docker-compose logs -f
kubectl logs -f -l app=cloudforge-ai
```

**Update Application:**
```powershell
helm upgrade cloudforge-ai ./helm-chart
kubectl rollout status deployment/backend
```