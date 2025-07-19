#!/bin/bash

# Hospital Financial Intelligence - Optimized Docker Build Script
# Supports both Mac and Windows environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="hospital-financial-ai"
TAG_OPTIMIZED="optimized"
TAG_DEV="dev"
PLATFORM="linux/amd64,linux/arm64"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to build optimized image
build_optimized() {
    print_status "Building optimized production image..."
    
    # Build with platform support for both Mac and Windows
    docker buildx build \
        --platform $PLATFORM \
        --file Dockerfile.optimized \
        --target runtime \
        --tag ${IMAGE_NAME}:${TAG_OPTIMIZED} \
        --tag ${IMAGE_NAME}:latest \
        --cache-from type=local,src=/tmp/.buildx-cache \
        --cache-to type=local,dest=/tmp/.buildx-cache \
        .
    
    print_success "Optimized image built successfully"
}

# Function to build development image
build_dev() {
    print_status "Building development image..."
    
    docker buildx build \
        --platform $PLATFORM \
        --file Dockerfile.optimized \
        --target runtime \
        --tag ${IMAGE_NAME}:${TAG_DEV} \
        --cache-from type=local,src=/tmp/.buildx-cache \
        --cache-to type=local,dest=/tmp/.buildx-cache \
        .
    
    print_success "Development image built successfully"
}

# Function to run the application
run_app() {
    local mode=${1:-production}
    
    if [ "$mode" = "dev" ]; then
        print_status "Starting development environment..."
        docker-compose -f docker-compose.optimized.yml --profile dev up --build
    else
        print_status "Starting production environment..."
        docker-compose -f docker-compose.optimized.yml up --build
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose -f docker-compose.optimized.yml down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Hospital Financial Intelligence - Docker Build Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build optimized production image"
    echo "  build-dev   Build development image"
    echo "  run         Run production environment"
    echo "  run-dev     Run development environment"
    echo "  clean       Clean up Docker resources"
    echo "  all         Build and run production environment"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run-dev"
    echo "  $0 all"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        "build")
            build_optimized
            ;;
        "build-dev")
            build_dev
            ;;
        "run")
            run_app "production"
            ;;
        "run-dev")
            run_app "dev"
            ;;
        "clean")
            cleanup
            ;;
        "all")
            build_optimized
            run_app "production"
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@" 