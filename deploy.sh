#!/bin/bash

# Predictive Maintenance System Deployment Script
# This script deploys the complete system including Kafka, API, and streaming components

set -e  # Exit on any error

echo "🚀 Starting Predictive Maintenance System Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if required files exist
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in current directory."
        exit 1
    fi
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory."
        exit 1
    fi
    
    if [ ! -f "serve_api.py" ]; then
        print_error "serve_api.py not found in current directory."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Check if model and data are available
check_model_data() {
    print_status "Checking model and data availability..."
    
    if [ ! -d "checkpoints" ]; then
        print_warning "checkpoints directory not found. You need to train a model first."
        print_warning "Run: python models/run_pipeline.py --skip_data_prep --skip_training"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    if [ ! -d "transformer_data" ]; then
        print_warning "transformer_data directory not found. You need to prepare data first."
        print_warning "Run: python models/run_pipeline.py --skip_training --skip_evaluation"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Model and data check completed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    docker-compose build
    
    if [ $? -eq 0 ]; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Start Kafka and Zookeeper first
    print_status "Starting Kafka and Zookeeper..."
    docker-compose up -d zookeeper kafka
    
    # Wait for Kafka to be ready
    print_status "Waiting for Kafka to be ready..."
    sleep 30
    
    # Start the API
    print_status "Starting Predictive Maintenance API..."
    docker-compose up -d predictive-maintenance-api
    
    # Wait for API to be ready
    print_status "Waiting for API to be ready..."
    sleep 20
    
    # Start streaming components
    print_status "Starting streaming components..."
    docker-compose up -d kafka-producer kafka-consumer
    
    # Start Kafka UI
    print_status "Starting Kafka UI..."
    docker-compose up -d kafka-ui
    
    print_success "All services started successfully"
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check if services are running
    services=("zookeeper" "kafka" "predictive-maintenance-api" "kafka-producer" "kafka-consumer" "kafka-ui")
    
    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            print_success "$service is running"
        else
            print_error "$service is not running"
        fi
    done
    
    # Check API health
    print_status "Checking API health..."
    if curl -s http://localhost:8000/ > /dev/null; then
        print_success "API is responding"
    else
        print_warning "API is not responding yet (may need more time to start)"
    fi
    
    # Check Kafka UI
    print_status "Checking Kafka UI..."
    if curl -s http://localhost:8080 > /dev/null; then
        print_success "Kafka UI is accessible"
    else
        print_warning "Kafka UI is not accessible yet (may need more time to start)"
    fi
}

# Display access information
show_access_info() {
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "====================================="
    echo ""
    echo "Access Information:"
    echo "-------------------"
    echo "📊 Predictive Maintenance API: http://localhost:8000"
    echo "📈 Kafka UI (Monitoring):     http://localhost:8080"
    echo "🔧 Kafka Broker:              localhost:9092"
    echo ""
    echo "API Endpoints:"
    echo "--------------"
    echo "GET  /                    - API information"
    echo "POST /predict             - RUL prediction"
    echo ""
    echo "Example API call:"
    echo "curl -X POST http://localhost:8000/predict \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"data\": [[0.1, 0.2, ...]], \"feature_names\": null}'"
    echo ""
    echo "Monitoring:"
    echo "-----------"
    echo "• View logs: docker-compose logs -f"
    echo "• Stop services: docker-compose down"
    echo "• Restart services: docker-compose restart"
    echo ""
    echo "Streaming Pipeline:"
    echo "-------------------"
    echo "• Kafka Producer: Simulating sensor data"
    echo "• Kafka Consumer: Processing data and generating alerts"
    echo "• Alerts will appear in the console for RUL < 25 cycles"
    echo ""
}

# Main deployment function
main() {
    echo "Predictive Maintenance System Deployment"
    echo "========================================"
    echo ""
    
    check_prerequisites
    check_model_data
    build_images
    start_services
    
    # Wait a bit for all services to stabilize
    print_status "Waiting for services to stabilize..."
    sleep 10
    
    check_health
    show_access_info
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping all services..."
        docker-compose down
        print_success "All services stopped"
        ;;
    "restart")
        print_status "Restarting all services..."
        docker-compose restart
        print_success "All services restarted"
        ;;
    "logs")
        print_status "Showing logs..."
        docker-compose logs -f
        ;;
    "status")
        print_status "Service status:"
        docker-compose ps
        ;;
    "clean")
        print_status "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    *)
        main
        ;;
esac 