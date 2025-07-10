#!/bin/bash
# Hospital Financial Intelligence - Docker Build Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
IMAGE_NAME="hospital-financial-ai"
CONTAINER_NAME="hospital-financial-intelligence"
VERSION="latest"

echo -e "${BLUE}🏥 Hospital Financial Intelligence - Docker Build Script${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Parse command line arguments
case "$1" in
    "build")
        print_status "Building Docker image..."
        docker build -t $IMAGE_NAME:$VERSION .
        print_status "Build completed successfully!"
        ;;
    
    "run")
        print_status "Starting Hospital Financial Intelligence container..."
        docker-compose up -d hospital-ai
        print_status "Container started! Dashboard available at: http://localhost:8502"
        ;;
    
    "dev")
        print_status "Starting development container..."
        docker-compose --profile dev up -d hospital-ai-dev
        print_status "Development container started! Dashboard available at: http://localhost:8503"
        ;;
    
    "stop")
        print_status "Stopping containers..."
        docker-compose down
        print_status "Containers stopped."
        ;;
    
    "rebuild")
        print_status "Rebuilding and restarting..."
        docker-compose down
        docker build -t $IMAGE_NAME:$VERSION .
        docker-compose up -d hospital-ai
        print_status "Rebuild completed! Dashboard available at: http://localhost:8502"
        ;;
    
    "logs")
        print_status "Showing container logs..."
        docker-compose logs -f hospital-ai
        ;;
    
    "shell")
        print_status "Opening shell in container..."
        docker exec -it $CONTAINER_NAME /bin/bash
        ;;
    
    "clean")
        print_warning "Cleaning up Docker resources..."
        docker-compose down
        docker rmi $IMAGE_NAME:$VERSION 2>/dev/null || true
        docker system prune -f
        print_status "Cleanup completed."
        ;;
    
    "help"|"")
        echo "Usage: $0 {build|run|dev|stop|rebuild|logs|shell|clean|help}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  run     - Start the production container"
        echo "  dev     - Start the development container"
        echo "  stop    - Stop all containers"
        echo "  rebuild - Rebuild and restart containers"
        echo "  logs    - Show container logs"
        echo "  shell   - Open shell in running container"
        echo "  clean   - Clean up Docker resources"
        echo "  help    - Show this help message"
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac 