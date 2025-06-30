# 🏥 Hospital Financial Intelligence - Docker Deployment

This document provides complete instructions for building, testing, and deploying the Hospital Financial Intelligence platform using Docker.

## Quick Start

### 1. Local Testing
```bash
# Test the Docker build locally
./docker-publish.sh test
```

### 2. Build for Production
```bash
# Build multi-platform image
./docker-publish.sh build
```

### 3. Publish to Docker Hub
```bash
# Set your Docker Hub repository
export DOCKER_HUB_REPO="yourusername/hospital-ai"

# Push to Docker Hub
./docker-publish.sh push
```

## Docker Hub Publishing

### Prerequisites
1. **Docker Desktop** installed and running
2. **Docker Hub account** (free at hub.docker.com)
3. **Docker buildx** enabled (included in Docker Desktop)

### Setup for Publishing

1. **Create Docker Hub Repository**
   ```bash
   # Login to Docker Hub
   docker login
   
   # Create repository at hub.docker.com
   # Repository name example: yourusername/hospital-financial-ai
   ```

2. **Set Environment Variable**
   ```bash
   export DOCKER_HUB_REPO="yourusername/hospital-financial-ai"
   ```

3. **Test → Build → Publish**
   ```bash
   # Test locally first
   ./docker-publish.sh test
   
   # Clean up test
   ./docker-publish.sh clean-test
   
   # Publish to Docker Hub
   ./docker-publish.sh push
   ```

## Running the Published Image

### From Docker Hub
```bash
# Pull and run the latest version
docker pull yourusername/hospital-financial-ai:latest
docker run -p 8502:8502 yourusername/hospital-financial-ai:latest
```

### Using Docker Compose (Recommended)
```bash
# Clone the repository or download docker-compose.yml
git clone https://github.com/yourusername/hospital-financial-ai.git
cd hospital-financial-ai

# Update docker-compose.yml with your image name
# Then run:
docker-compose up -d
```

## Advanced Usage

### Custom Configuration
```bash
# Run with custom port
docker run -p 9000:8502 yourusername/hospital-financial-ai:latest

# Run with volume mounts for data persistence
docker run -p 8502:8502 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  yourusername/hospital-financial-ai:latest
```

### Development Mode
```bash
# Build for development
docker build -t hospital-ai:dev .

# Run with code mount for live editing
docker run -p 8502:8502 \
  -v $(pwd):/app \
  hospital-ai:dev
```

## Platform Support

This Docker image supports multiple platforms:
- **linux/amd64** (Intel/AMD x64)
- **linux/arm64** (Apple Silicon, ARM servers)

The build process automatically creates multi-platform images that work on:
- ✅ Intel/AMD Mac computers
- ✅ Apple Silicon Mac computers (M1/M2/M3)
- ✅ Linux servers (x64)
- ✅ ARM-based Linux servers
- ✅ Windows with Docker Desktop

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8502` | Port for the web interface |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Server bind address |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable analytics |

## Troubleshooting

### Build Issues

**Problem**: Dependencies fail to install
```bash
# Solution: Clean build without cache
docker build --no-cache -t hospital-ai .
```

**Problem**: Platform not supported
```bash
# Solution: Build for specific platform
docker buildx build --platform linux/amd64 -t hospital-ai .
```

### Runtime Issues

**Problem**: Application not accessible
```bash
# Check if container is running
docker ps

# Check application logs
docker logs [container-id]

# Check port binding
docker port [container-id]
```

**Problem**: Permission denied errors
```bash
# The image runs as non-root user 'appuser'
# Ensure mounted volumes have proper permissions
chmod -R 755 data/ models/ reports/
```

### Health Check
```bash
# Test health endpoint
curl http://localhost:8502/_stcore/health

# Expected response: {"status": "ok"}
```

## Security Considerations

1. **Non-root User**: Container runs as `appuser` (not root)
2. **Read-only Filesystem**: Consider `--read-only` for production
3. **Resource Limits**: Set memory/CPU limits in production
4. **Network Security**: Use proper firewall rules for port 8502

## Production Deployment Examples

### Docker Swarm
```yaml
version: '3.8'
services:
  hospital-ai:
    image: yourusername/hospital-financial-ai:latest
    ports:
      - "8502:8502"
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8502/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hospital-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hospital-ai
  template:
    metadata:
      labels:
        app: hospital-ai
    spec:
      containers:
      - name: hospital-ai
        image: yourusername/hospital-financial-ai:latest
        ports:
        - containerPort: 8502
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8502
          initialDelaySeconds: 60
          periodSeconds: 30
```

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `docker-build.sh` | Basic build operations | `./docker-build.sh build` |
| `docker-publish.sh` | Advanced publishing | `./docker-publish.sh help` |
| `docker-compose.yml` | Local development | `docker-compose up` |

## Data Persistence

For production use, mount these directories:
- `/app/data` - Input data and features
- `/app/models` - Trained ML models
- `/app/reports` - Generated reports
- `/app/visuals` - Charts and visualizations
- `/app/logs` - Application logs

Example with persistence:
```bash
docker run -d \
  --name hospital-ai \
  -p 8502:8502 \
  -v hospital_data:/app/data \
  -v hospital_models:/app/models \
  -v hospital_reports:/app/reports \
  yourusername/hospital-financial-ai:latest
```

## Support

For Docker-related issues:
1. Check the logs: `docker logs [container-name]`
2. Verify the image: `docker images | grep hospital`
3. Test health: `curl localhost:8502/_stcore/health`
4. Report issues with Docker version and platform info

---

**Ready to deploy your Hospital Financial Intelligence platform anywhere!** 🚀 