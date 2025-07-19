#!/bin/bash

# Hospital Financial Intelligence - Docker Entrypoint
# Handles different startup modes and ensures proper container lifecycle

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
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

# Function to validate environment
validate_environment() {
    print_info "🔧 Validating environment setup..."
    
    # Check if required files exist
    if [[ ! -f "streamlit_dashboard_modern.py" ]]; then
        print_error "Streamlit dashboard file not found"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p data/raw data/processed data/features data/features_enhanced \
        models reports visuals logs
    
    print_success "Environment validation completed"
}

# Function to launch dashboard
launch_dashboard() {
    local port=${1:-8502}
    local address=${2:-"0.0.0.0"}
    
    print_info "🚀 Launching Hospital Financial Intelligence Dashboard"
    print_info "🌐 URL: http://localhost:${port}"
    print_info "📊 Dashboard: Professional healthcare analytics interface"
    
    # Launch Streamlit with proper configuration
    exec streamlit run streamlit_dashboard_modern.py \
        --server.port "${port}" \
        --server.address "${address}" \
        --server.headless true \
        --browser.gatherUsageStats false
}

# Function to run pipeline
run_pipeline() {
    print_info "🔧 Running Hospital Financial Intelligence Pipeline"
    
    # Run the pipeline orchestrator
    exec python pipeline.py "$@"
}

# Function to show help
show_help() {
    echo "Hospital Financial Intelligence - Docker Entrypoint"
    echo ""
    echo "Usage:"
    echo "  docker-entrypoint.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  dashboard [PORT]     Launch dashboard only (default: 8502)"
    echo "  pipeline [ARGS]      Run pipeline with arguments"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  docker-entrypoint.sh dashboard 8503"
    echo "  docker-entrypoint.sh pipeline --dashboard"
    echo "  docker-entrypoint.sh pipeline --full"
    echo ""
    echo "Environment Variables:"
    echo "  STREAMLIT_SERVER_PORT     Dashboard port (default: 8502)"
    echo "  STREAMLIT_SERVER_ADDRESS  Dashboard address (default: 0.0.0.0)"
    echo "  PYTHONUNBUFFERED          Python output buffering (default: 1)"
}

# Main execution
main() {
    # Validate environment first
    validate_environment
    
    # Parse command
    case "${1:-dashboard}" in
        "dashboard")
            local port=${2:-${STREAMLIT_SERVER_PORT:-8502}}
            local address=${STREAMLIT_SERVER_ADDRESS:-"0.0.0.0"}
            launch_dashboard "${port}" "${address}"
            ;;
        "pipeline")
            shift  # Remove 'pipeline' from arguments
            run_pipeline "$@"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_warning "Unknown command: $1"
            print_info "Defaulting to dashboard mode"
            launch_dashboard
            ;;
    esac
}

# Run main function with all arguments
main "$@" 