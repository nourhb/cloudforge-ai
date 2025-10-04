# CloudForge AI - Oracle Cloud Free Tier Deployment Guide
## Production-Ready Deployment on Oracle Cloud Infrastructure

### Oracle Cloud Free Tier Specifications
```
Available Resources (Always Free):
├─ Compute Instances: 2x VM.Standard.A1.Flex
│  ├─ CPU: 4 OCPU total (ARM-based Ampere A1)
│  ├─ Memory: 24 GB total
│  ├─ Network: 10 Gbps
│  └─ Operating System: Oracle Linux 8.8 / Ubuntu 22.04
├─ Block Storage: 200 GB total
├─ Load Balancer: 1 instance (10 Mbps)
├─ VCN: 1 Virtual Cloud Network
└─ Object Storage: 20 GB
```

### Resource Allocation Strategy
```yaml
# Optimal resource distribution for CloudForge AI
Instance 1 (Control Plane + Database):
  cpu: 2 OCPU
  memory: 12 GB
  storage: 100 GB
  services:
    - Kubernetes Control Plane
    - PostgreSQL Database
    - Redis Cache
    - Prometheus Monitoring

Instance 2 (Worker Node + Services):
  cpu: 2 OCPU  
  memory: 12 GB
  storage: 100 GB
  services:
    - Kubernetes Worker Node
    - CloudForge Backend
    - CloudForge Frontend
    - AI Services
    - MinIO Object Storage
```

### Terraform Infrastructure as Code
```hcl
# terraform/oracle-cloud/main.tf
terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.17.0"
    }
  }
}

provider "oci" {
  region           = var.region
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
}

# Virtual Cloud Network
resource "oci_core_vcn" "cloudforge_vcn" {
  compartment_id = var.compartment_id
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "CloudForge-VCN"
  dns_label      = "cloudforge"
  
  freeform_tags = {
    Project     = "CloudForge-AI"
    Environment = "Production"
    Version     = "2.0.0"
  }
}

# Internet Gateway
resource "oci_core_internet_gateway" "cloudforge_igw" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.cloudforge_vcn.id
  display_name   = "CloudForge-IGW"
  enabled        = true
}

# Public Subnet for Load Balancer
resource "oci_core_subnet" "public_subnet" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.cloudforge_vcn.id
  cidr_block     = "10.0.1.0/24"
  display_name   = "CloudForge-Public-Subnet"
  dns_label      = "public"
  
  route_table_id = oci_core_route_table.public_route_table.id
  security_list_ids = [oci_core_security_list.public_security_list.id]
}

# Private Subnet for Kubernetes Nodes  
resource "oci_core_subnet" "private_subnet" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.cloudforge_vcn.id
  cidr_block     = "10.0.2.0/24"
  display_name   = "CloudForge-Private-Subnet"
  dns_label      = "private"
  
  prohibit_public_ip_on_vnic = true
  route_table_id = oci_core_route_table.private_route_table.id
  security_list_ids = [oci_core_security_list.private_security_list.id]
}

# NAT Gateway for private subnet internet access
resource "oci_core_nat_gateway" "cloudforge_nat" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.cloudforge_vcn.id
  display_name   = "CloudForge-NAT"
}

# Control Plane Instance
resource "oci_core_instance" "k8s_control_plane" {
  compartment_id      = var.compartment_id
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "CloudForge-Control-Plane"
  shape               = "VM.Standard.A1.Flex"
  
  shape_config {
    ocpus         = 2
    memory_in_gbs = 12
  }
  
  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_images.images[0].id
  }
  
  create_vnic_details {
    subnet_id                 = oci_core_subnet.private_subnet.id
    display_name              = "CloudForge-Control-VNIC"
    assign_public_ip          = false
    assign_private_dns_record = true
    hostname_label            = "control-plane"
  }
  
  metadata = {
    ssh_authorized_keys = file(var.ssh_public_key_path)
    user_data = base64encode(templatefile("${path.module}/cloud-init/control-plane.yaml", {
      k8s_version = "1.28.2"
      pod_subnet  = "10.244.0.0/16"
    }))
  }
  
  freeform_tags = {
    Role        = "ControlPlane"
    Project     = "CloudForge-AI"
    Environment = "Production"
  }
}

# Worker Node Instance
resource "oci_core_instance" "k8s_worker" {
  compartment_id      = var.compartment_id
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "CloudForge-Worker"
  shape               = "VM.Standard.A1.Flex"
  
  shape_config {
    ocpus         = 2
    memory_in_gbs = 12
  }
  
  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_images.images[0].id
  }
  
  create_vnic_details {
    subnet_id                 = oci_core_subnet.private_subnet.id
    display_name              = "CloudForge-Worker-VNIC"
    assign_public_ip          = false
    assign_private_dns_record = true
    hostname_label            = "worker-1"
  }
  
  metadata = {
    ssh_authorized_keys = file(var.ssh_public_key_path)
    user_data = base64encode(templatefile("${path.module}/cloud-init/worker.yaml", {
      control_plane_ip = oci_core_instance.k8s_control_plane.private_ip
      k8s_version     = "1.28.2"
    }))
  }
  
  freeform_tags = {
    Role        = "Worker"
    Project     = "CloudForge-AI"
    Environment = "Production"
  }
}

# Block Storage for persistent data
resource "oci_core_volume" "cloudforge_data" {
  compartment_id      = var.compartment_id
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "CloudForge-Data-Volume"
  size_in_gbs         = 100
  
  freeform_tags = {
    Purpose     = "PersistentData"
    Project     = "CloudForge-AI"
    Environment = "Production"
  }
}

# Load Balancer for external access
resource "oci_load_balancer_load_balancer" "cloudforge_lb" {
  compartment_id = var.compartment_id
  display_name   = "CloudForge-LoadBalancer"
  shape          = "flexible"
  
  shape_details {
    minimum_bandwidth_in_mbps = 10
    maximum_bandwidth_in_mbps = 10
  }
  
  subnet_ids = [oci_core_subnet.public_subnet.id]
  
  is_private = false
  
  freeform_tags = {
    Component   = "LoadBalancer"
    Project     = "CloudForge-AI"
    Environment = "Production"
  }
}
```

### Cloud-Init Configuration for Automated Setup
```yaml
# cloud-init/control-plane.yaml
#cloud-config
package_update: true
package_upgrade: true

packages:
  - docker.io
  - curl
  - wget
  - git
  - htop
  - net-tools

runcmd:
  # Install Kubernetes
  - curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg
  - echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
  - apt-get update
  - apt-get install -y kubelet=1.28.2-00 kubeadm=1.28.2-00 kubectl=1.28.2-00
  - apt-mark hold kubelet kubeadm kubectl
  
  # Configure Docker
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker ubuntu
  
  # Configure kernel modules
  - modprobe overlay
  - modprobe br_netfilter
  - echo 'overlay' >> /etc/modules-load.d/containerd.conf
  - echo 'br_netfilter' >> /etc/modules-load.d/containerd.conf
  
  # Configure sysctl
  - echo 'net.bridge.bridge-nf-call-iptables = 1' >> /etc/sysctl.d/99-kubernetes-cri.conf
  - echo 'net.ipv4.ip_forward = 1' >> /etc/sysctl.d/99-kubernetes-cri.conf
  - echo 'net.bridge.bridge-nf-call-ip6tables = 1' >> /etc/sysctl.d/99-kubernetes-cri.conf
  - sysctl --system
  
  # Initialize Kubernetes cluster
  - kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=$(hostname -I | awk '{print $1}')
  
  # Configure kubectl for ubuntu user
  - mkdir -p /home/ubuntu/.kube
  - cp -i /etc/kubernetes/admin.conf /home/ubuntu/.kube/config
  - chown ubuntu:ubuntu /home/ubuntu/.kube/config
  
  # Install Flannel CNI
  - sudo -u ubuntu kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
  
  # Install Helm
  - curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
  - echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
  - apt-get update
  - apt-get install helm
  
  # Create join command for worker nodes
  - kubeadm token create --print-join-command > /tmp/kubeadm-join.sh
  - chmod +x /tmp/kubeadm-join.sh

write_files:
  - path: /etc/docker/daemon.json
    content: |
      {
        "exec-opts": ["native.cgroupdriver=systemd"],
        "log-driver": "json-file",
        "log-opts": {
          "max-size": "100m"
        },
        "storage-driver": "overlay2"
      }
    permissions: '0644'
```

### Kubernetes Deployment Manifests
```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cloudforge-prod
  labels:
    name: cloudforge-prod
    environment: production
    version: "2.0.0"

---
# k8s/production/postgres-statefulset.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: cloudforge-prod
type: Opaque
data:
  postgres-password: Y2xvdWRmb3JnZS1wcm9kLXBhc3N3b3Jk  # base64: cloudforge-prod-password
  postgres-user: Y2xvdWRmb3JnZQ==  # base64: cloudforge

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: cloudforge-prod
data:
  POSTGRES_DB: "cloudforge_prod"
  POSTGRES_USER: "cloudforge"
  PGDATA: "/var/lib/postgresql/data/pgdata"

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: cloudforge-prod
spec:
  serviceName: postgres-headless
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      securityContext:
        runAsUser: 999
        runAsGroup: 999
        fsGroup: 999
      containers:
      - name: postgres
        image: postgres:14.9-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_USER
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: postgres-password
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: POSTGRES_DB
        - name: PGDATA
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: PGDATA
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config-volume
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - cloudforge
            - -d
            - cloudforge_prod
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - cloudforge
            - -d
            - cloudforge_prod
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-config-volume
        configMap:
          name: postgres-performance-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi

---
# PostgreSQL Performance Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-performance-config
  namespace: cloudforge-prod
data:
  postgresql.conf: |
    # Memory Configuration for 2GB limit
    shared_buffers = 512MB
    effective_cache_size = 1536MB
    maintenance_work_mem = 128MB
    work_mem = 32MB
    
    # Connection Settings
    max_connections = 100
    
    # Write-Ahead Logging
    wal_buffers = 16MB
    checkpoint_completion_target = 0.9
    
    # Query Planning
    random_page_cost = 1.1
    effective_io_concurrency = 200
    
    # Logging
    log_statement = 'all'
    log_duration = on
    log_min_duration_statement = 1000
    
    # Performance Monitoring
    track_activities = on
    track_counts = on
    track_functions = all
    track_io_timing = on

---
# k8s/production/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: cloudforge-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --appendonly
        - "yes"
        - --maxmemory
        - "1gb"
        - --maxmemory-policy
        - "allkeys-lru"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-data
        emptyDir: {}

---
# k8s/production/cloudforge-backend.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cloudforge-backend-secret
  namespace: cloudforge-prod
type: Opaque
data:
  jwt-secret: Y2xvdWRmb3JnZS1qd3Qtc2VjcmV0LXByb2QtMjAyNQ==  # cloudforge-jwt-secret-prod-2025
  database-url: cG9zdGdyZXNxbDovL2Nsb3VkZm9yZ2U6Y2xvdWRmb3JnZS1wcm9kLXBhc3N3b3JkQHBvc3RncmVzOjU0MzIvY2xvdWRmb3JnZV9wcm9k

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudforge-backend
  namespace: cloudforge-prod
  labels:
    app: cloudforge-backend
    version: "2.0.0"
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: cloudforge-backend
  template:
    metadata:
      labels:
        app: cloudforge-backend
        version: "2.0.0"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3001"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: backend
        image: cloudforge/backend:2.0.0
        ports:
        - containerPort: 3001
          name: http
        env:
        - name: NODE_ENV
          value: "production"
        - name: PORT
          value: "3001"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cloudforge-backend-secret
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: cloudforge-backend-secret
              key: jwt-secret
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: logs
        emptyDir: {}
      - name: tmp
        emptyDir: {}

---
# k8s/production/cloudforge-frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudforge-frontend
  namespace: cloudforge-prod
  labels:
    app: cloudforge-frontend
    version: "2.0.0"
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: cloudforge-frontend
  template:
    metadata:
      labels:
        app: cloudforge-frontend
        version: "2.0.0"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: frontend
        image: cloudforge/frontend:2.0.0
        ports:
        - containerPort: 3002
          name: http
        env:
        - name: NODE_ENV
          value: "production"
        - name: NEXT_PUBLIC_API_URL
          value: "https://api.cloudforge.local"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3002
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/production/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: cloudforge-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: cloudforge-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None

---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: cloudforge-prod
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: cloudforge-backend
  namespace: cloudforge-prod
  labels:
    app: cloudforge-backend
spec:
  selector:
    app: cloudforge-backend
  ports:
  - port: 80
    targetPort: 3001
    protocol: TCP
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: cloudforge-frontend
  namespace: cloudforge-prod
  labels:
    app: cloudforge-frontend
spec:
  selector:
    app: cloudforge-frontend
  ports:
  - port: 80
    targetPort: 3002
    protocol: TCP
  type: ClusterIP

---
# k8s/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cloudforge-ingress
  namespace: cloudforge-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - cloudforge.example.com
    - api.cloudforge.example.com
    secretName: cloudforge-tls
  rules:
  - host: cloudforge.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cloudforge-frontend
            port:
              number: 80
  - host: api.cloudforge.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cloudforge-backend
            port:
              number: 80

---
# k8s/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cloudforge-backend-hpa
  namespace: cloudforge-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cloudforge-backend
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cloudforge-frontend-hpa
  namespace: cloudforge-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cloudforge-frontend
  minReplicas: 2
  maxReplicas: 6
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Cost Optimization Configuration
```yaml
# k8s/cost-optimization/resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: cloudforge-quota
  namespace: cloudforge-prod
spec:
  hard:
    requests.cpu: "3"
    requests.memory: "6Gi"
    limits.cpu: "6" 
    limits.memory: "12Gi"
    pods: "20"
    persistentvolumeclaims: "5"
    services: "10"
    secrets: "10"
    configmaps: "20"

---
# Limit Ranges for cost control
apiVersion: v1
kind: LimitRange
metadata:
  name: cloudforge-limits
  namespace: cloudforge-prod
spec:
  limits:
  - type: Container
    default:
      cpu: "500m"
      memory: "512Mi"
    defaultRequest:
      cpu: "100m"
      memory: "128Mi"
    max:
      cpu: "2"
      memory: "2Gi"
    min:
      cpu: "50m"
      memory: "64Mi"
  - type: PersistentVolumeClaim
    max:
      storage: "100Gi"
    min:
      storage: "1Gi"
```

### Monitoring and Observability Stack
```yaml
# k8s/monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: cloudforge-system
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
      
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https
      
      - job_name: 'kubernetes-nodes'
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        kubernetes_sd_configs:
        - role: node
        relabel_configs:
        - action: labelmap
          regex: __meta_kubernetes_node_label_(.+)
      
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__
        - action: labelmap
          regex: __meta_kubernetes_pod_label_(.+)
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_pod_name]
          action: replace
          target_label: kubernetes_pod_name
      
      - job_name: 'cloudforge-backend'
        static_configs:
        - targets: ['cloudforge-backend.cloudforge-prod.svc.cluster.local:80']
        metrics_path: '/metrics'
        scrape_interval: 10s
      
      - job_name: 'postgres-exporter'
        static_configs:
        - targets: ['postgres-exporter.cloudforge-prod.svc.cluster.local:9187']

    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager.cloudforge-system.svc.cluster.local:9093

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: cloudforge-system
data:
  cloudforge-rules.yml: |
    groups:
    - name: cloudforge.rules
      rules:
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 2 minutes"
      
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 2 minutes"
      
      - alert: PodCrashLooping
        expr: increase(kube_pod_container_status_restarts_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is crash looping"
      
      - alert: DatabaseConnectionFailure
        expr: postgres_up == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is not responding"
      
      - alert: BackendServiceDown
        expr: up{job="cloudforge-backend"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Backend service is down"
          description: "CloudForge backend service is not responding"
```