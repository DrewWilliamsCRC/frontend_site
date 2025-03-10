#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Function to check if a container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to get container status
get_container_status() {
    local name=$1
    if container_running "$name"; then
        echo "running"
    elif container_exists "$name"; then
        echo "exists"
    else
        echo "none"
    fi
}

# Function to prompt for AI server rebuild
prompt_ai_server_rebuild() {
    local ai_status=$(get_container_status "frontend_site-ai_server-1")
    
    if [ "$ai_status" = "running" ]; then
        echo -e "${YELLOW}AI server is currently running.${NC}"
        read -p "Do you want to rebuild the AI server? This may take several minutes. (y/N): " rebuild
        if [[ $rebuild =~ ^[Yy]$ ]]; then
            return 0
        else
            return 1
        fi
    elif [ "$ai_status" = "exists" ]; then
        echo -e "${YELLOW}AI server container exists but is not running.${NC}"
        read -p "Do you want to rebuild the AI server? This may take several minutes. (y/N): " rebuild
        if [[ $rebuild =~ ^[Yy]$ ]]; then
            return 0
        else
            docker-compose start ai_server
            return 1
        fi
    else
        echo -e "${YELLOW}AI server container does not exist. Will build for the first time.${NC}"
        return 0
    fi
}

case "$1" in
    "up")
        if prompt_ai_server_rebuild; then
            echo -e "${GREEN}Starting all services with AI server rebuild...${NC}"
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
        else
            echo -e "${GREEN}Starting services without rebuilding AI server...${NC}"
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --no-build ai_server --build frontend db
        fi
        ;;
    "down")
        read -p "Do you want to stop the AI server as well? (y/N): " stop_ai
        if [[ $stop_ai =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Stopping all services...${NC}"
            docker-compose down
        else
            echo -e "${GREEN}Stopping frontend and db, keeping AI server running...${NC}"
            docker-compose stop frontend db
        fi
        ;;
    "restart-ui")
        echo -e "${GREEN}Restarting only frontend service...${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build frontend
        ;;
    *)
        echo "Usage: $0 {up|down|restart-ui}"
        echo "  up          - Start the development environment"
        echo "  down        - Stop the development environment"
        echo "  restart-ui  - Rebuild and restart only the frontend service"
        exit 1
        ;;
esac 