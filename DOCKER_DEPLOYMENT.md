# Hospital Financial Intelligence - Docker Deployment Guide

## 🚀 Quick Start with Docker Hub

### **Pull and Run from Docker Hub**

```bash
# Pull the latest optimized image
docker pull esengendo730/hospital-financial-ai:latest

# Run the dashboard
docker run -d --name hospital-ai -p 8502:8502 esengendo730/hospital-financial-ai:latest

# Access dashboard
open http://localhost:8502
```

## 📋 Available Tags

- **`latest`** - Latest optimized production build
- **`optimized`** - Specifically tagged optimized build

## 🏥 Full Production Deployment

### **With Sample Data (Quick Demo)**

```bash
# Pull and run with basic functionality
docker run -d \
  --name hospital-financial-intelligence \
  -p 8502:8502 \
  --restart unless-stopped \
  esengendo730/hospital-financial-ai:latest
```

### **With Your Own Data**

```bash
# Run with your data directories mounted
docker run -d \
  --name hospital-financial-intelligence \
  -p 8502:8502 \
  -v /path/to/your/data:/app/data \
  -v /path/to/your/models:/app/models \
  -v /path/to/your/reports:/app/reports \
  --restart unless-stopped \
  esengendo730/hospital-financial-ai:latest
```

## 🐳 Docker Compose Deployment

### **Create `docker-compose.yml`**

```yaml
version: '3.8'

services:
  hospital-ai:
    image: esengendo730/hospital-financial-ai:latest
    container_name: hospital-financial-intelligence
    ports:
      - "8502:8502"
    environment:
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_PORT=8502
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8502/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### **Deploy**

```bash
# Start the service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔧 Advanced Configuration

### **Environment Variables**

```bash
docker run -d \
  --name hospital-ai \
  -p 8502:8502 \
  -e STREAMLIT_SERVER_PORT=8502 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  esengendo730/hospital-financial-ai:latest
```

### **Resource Limits**

```bash
docker run -d \
  --name hospital-ai \
  -p 8502:8502 \
  --memory=2g \
  --cpus=1.0 \
  esengendo730/hospital-financial-ai:latest
```

## 🌐 Cloud Deployment

### **AWS ECS**

```json
{
  "family": "hospital-financial-ai",
  "containerDefinitions": [
    {
      "name": "hospital-ai",
      "image": "esengendo730/hospital-financial-ai:latest",
      "portMappings": [
        {
          "containerPort": 8502,
          "protocol": "tcp"
        }
      ],
      "memory": 2048,
      "cpu": 1024
    }
  ]
}
```

### **Google Cloud Run**

```bash
# Deploy to Cloud Run
gcloud run deploy hospital-financial-ai \
  --image esengendo730/hospital-financial-ai:latest \
  --platform managed \
  --port 8502 \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated
```

### **Azure Container Instances**

```bash
# Deploy to Azure
az container create \
  --resource-group myResourceGroup \
  --name hospital-financial-ai \
  --image esengendo730/hospital-financial-ai:latest \
  --ports 8502 \
  --memory 2 \
  --cpu 1
```

## 📊 Features Included

- ✅ **439 Real Hospitals** with official names
- ✅ **21 Years of Data** (2003-2023)
- ✅ **147 Financial Features** per hospital
- ✅ **AI-Powered Analysis** with Groq integration
- ✅ **Interactive Dashboard** with modern UI
- ✅ **Health Monitoring** and auto-restart
- ✅ **Security Hardened** (non-root user)
- ✅ **Cross-Platform** (Mac/Windows/Linux)

## 🔍 Management Commands

```bash
# Check container status
docker ps

# View logs
docker logs hospital-financial-intelligence

# Stop container
docker stop hospital-financial-intelligence

# Start container
docker start hospital-financial-intelligence

# Remove container
docker rm hospital-financial-intelligence

# Update to latest
docker pull esengendo730/hospital-financial-ai:latest
docker stop hospital-financial-intelligence
docker rm hospital-financial-intelligence
docker run -d --name hospital-financial-intelligence -p 8502:8502 esengendo730/hospital-financial-ai:latest
```

## 🆘 Troubleshooting

### **Port Already in Use**

```bash
# Find what's using port 8502
lsof -i :8502

# Use different port
docker run -d --name hospital-ai -p 8503:8502 esengendo730/hospital-financial-ai:latest
```

### **Container Won't Start**

```bash
# Check logs
docker logs hospital-financial-intelligence

# Check container health
docker inspect hospital-financial-intelligence --format='{{.State.Health.Status}}'
```

### **Performance Issues**

```bash
# Increase memory limit
docker run -d --name hospital-ai -p 8502:8502 --memory=4g esengendo730/hospital-financial-ai:latest
```

## 📈 Production Recommendations

1. **Resource Allocation**: Minimum 2GB RAM, 1 CPU core
2. **Health Monitoring**: Use built-in health checks
3. **Log Management**: Set up log rotation and monitoring
4. **Backup Strategy**: Regular backups of data volumes
5. **Security**: Run behind reverse proxy (nginx/traefik)
6. **SSL/TLS**: Enable HTTPS for production deployments

## 🔗 Repository Information

- **Docker Hub**: https://hub.docker.com/r/esengendo730/hospital-financial-ai
- **Image Size**: ~1.4GB (optimized multi-stage build)
- **Base Image**: Python 3.10 slim
- **Platform**: linux/amd64, linux/arm64
- **Last Updated**: $(date)

---

**Ready to deploy? Just run:**

```bash
docker run -d --name hospital-ai -p 8502:8502 esengendo730/hospital-financial-ai:latest
```

**Then visit: http://localhost:8502** 🚀 