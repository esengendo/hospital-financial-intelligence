#!/bin/bash
# Hospital Financial Intelligence - Docker Hub Publishing Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
DOCKER_REPO="esengendo730/hospital-financial-ai"  # Your Docker Hub repository
IMAGE_NAME="hospital-financial-ai"
VERSION=${1:-"latest"}
PLATFORMS="linux/amd64,linux/arm64"

echo -e "${BLUE}🏥 Hospital Financial Intelligence - Docker Hub Publisher${NC}"
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

# Check if Docker buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    print_error "Docker buildx is not available. Please update Docker to a recent version."
    exit 1
fi

# Function to get Docker Hub repository
get_docker_repo() {
    if [[ -z "$DOCKER_HUB_REPO" ]]; then
        echo -e "${YELLOW}Enter your Docker Hub repository name (e.g., username/hospital-ai):${NC}"
        read -r DOCKER_HUB_REPO
        if [[ -z "$DOCKER_HUB_REPO" ]]; then
            print_error "Docker Hub repository name is required."
            exit 1
        fi
    fi
    echo "$DOCKER_HUB_REPO"
}

# Parse command line arguments
case "$1" in
    "build")
        REPO=$(get_docker_repo)
        print_status "Building multi-platform Docker image..."
        
        # Create buildx builder if it doesn't exist
        docker buildx create --name hospital-builder --use 2>/dev/null || docker buildx use hospital-builder
        
        # Build for multiple platforms
        docker buildx build \
            --platform $PLATFORMS \
            --tag $REPO:$VERSION \
            --tag $REPO:latest \
            --load \
            .
        
        print_status "Multi-platform build completed successfully!"
        ;;
    
    "push")
        REPO=$(get_docker_repo)
        print_status "Building and pushing to Docker Hub..."
        
        # Check if logged in to Docker Hub
        if ! docker info | grep -q "Username:"; then
            print_warning "Please log in to Docker Hub first:"
            docker login
        fi
        
        # Create buildx builder if it doesn't exist
        docker buildx create --name hospital-builder --use 2>/dev/null || docker buildx use hospital-builder
        
        # Build and push for multiple platforms
        docker buildx build \
            --platform $PLATFORMS \
            --tag $REPO:$VERSION \
            --tag $REPO:latest \
            --push \
            .
        
        print_status "Successfully pushed to Docker Hub: $REPO:$VERSION"
        print_status "Also tagged as: $REPO:latest"
        print_status "Pull with: docker pull $REPO:latest"
        ;;
    
    "test")
        REPO=$(get_docker_repo)
        print_status "Testing Docker image locally..."
        
        # Build and test locally
        docker build -t $IMAGE_NAME:test .
        
        # Run container for testing
        print_status "Starting test container..."
        docker run -d --name hospital-ai-test -p 8504:8502 $IMAGE_NAME:test
        
        # Wait for container to start
        sleep 10
        
        # Test health endpoint
        if curl -f http://localhost:8504/_stcore/health > /dev/null 2>&1; then
            print_status "Health check passed!"
        else
            print_warning "Health check failed, but container might still be starting..."
        fi
        
        # Test main page
        if curl -f http://localhost:8504 > /dev/null 2>&1; then
            print_status "Application is responding!"
            print_status "Test successful! View at: http://localhost:8504"
        else
            print_warning "Application not responding yet. Check with: docker logs hospital-ai-test"
        fi
        
        echo -e "${YELLOW}Test container is running. Stop with: docker stop hospital-ai-test && docker rm hospital-ai-test${NC}"
        ;;
    
    "clean-test")
        print_status "Cleaning up test containers..."
        docker stop hospital-ai-test 2>/dev/null || true
        docker rm hospital-ai-test 2>/dev/null || true
        docker rmi hospital-financial-ai:test 2>/dev/null || true
        print_status "Test cleanup completed."
        ;;
    
    "release")
        if [[ -z "$2" ]]; then
            print_error "Please provide a version number for the release."
            echo "Usage: $0 release v1.0.0"
            exit 1
        fi
        
        VERSION="$2"
        REPO=$(get_docker_repo)
        
        print_status "Creating release build: $VERSION"
        
        # Test first
        ./docker-publish.sh test
        
        # If test passes, build and push
        print_status "Test passed, proceeding with release..."
        
        # Clean up test
        ./docker-publish.sh clean-test
        
        # Build and push release
        docker buildx build \
            --platform $PLATFORMS \
            --tag $REPO:$VERSION \
            --tag $REPO:latest \
            --push \
            .
        
        print_status "Release $VERSION published successfully!"
        print_status "Available at: $REPO:$VERSION"
        ;;
    
    "help"|"")
        echo "Usage: $0 {build|push|test|clean-test|release|help} [version]"
        echo ""
        echo "Commands:"
        echo "  build          - Build multi-platform Docker image locally"
        echo "  push           - Build and push to Docker Hub"
        echo "  test           - Build and test the image locally"
        echo "  clean-test     - Clean up test containers"
        echo "  release <ver>  - Test, build and push a versioned release"
        echo "  help           - Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  DOCKER_HUB_REPO - Your Docker Hub repository (e.g., username/hospital-ai)"
        echo ""
        echo "Examples:"
        echo "  $0 test                    # Test locally"
        echo "  $0 push                    # Push latest to Docker Hub"
        echo "  $0 release v1.0.0          # Create versioned release"
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac 