#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in your PATH.${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running.${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: docker-compose command not found. Will try using 'docker compose' instead.${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Make sure .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found.${NC}"
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo -e "${YELLOW}Please update the .env file with your actual configuration values.${NC}"
    else
        echo -e "${RED}Error: Neither .env nor .env.example found.${NC}"
        echo "Please create a .env file with your configuration variables."
        exit 1
    fi
fi

# Check and correct architecture settings in .env
check_architecture_settings() {
    # Check if any ARM-related settings exist
    if grep -q "TARGETARCH=arm\|BUILDPLATFORM=.*arm\|TARGETPLATFORM=.*arm" .env; then
        echo -e "${YELLOW}Warning: ARM architecture settings found in .env file.${NC}"
        echo "Updating to use amd64 architecture..."
        
        # Replace ARM settings with amd64
        sed -i.bak 's/TARGETARCH=arm.*/TARGETARCH=amd64/g' .env
        sed -i.bak 's/BUILDPLATFORM=.*arm.*/BUILDPLATFORM=linux\/amd64/g' .env
        sed -i.bak 's/TARGETPLATFORM=.*arm.*/TARGETPLATFORM=linux\/amd64/g' .env
        
        # Add settings if they don't exist
        if ! grep -q "TARGETARCH=" .env; then
            echo "TARGETARCH=amd64" >> .env
        fi
        if ! grep -q "BUILDPLATFORM=" .env; then
            echo "BUILDPLATFORM=linux/amd64" >> .env
        fi
        if ! grep -q "TARGETPLATFORM=" .env; then
            echo "TARGETPLATFORM=linux/amd64" >> .env
        fi
        
        echo -e "${GREEN}Architecture settings updated to amd64.${NC}"
    else
        # Add settings if they don't exist
        local updated=false
        if ! grep -q "TARGETARCH=" .env; then
            echo "TARGETARCH=amd64" >> .env
            updated=true
        fi
        if ! grep -q "BUILDPLATFORM=" .env; then
            echo "BUILDPLATFORM=linux/amd64" >> .env
            updated=true
        fi
        if ! grep -q "TARGETPLATFORM=" .env; then
            echo "TARGETPLATFORM=linux/amd64" >> .env
            updated=true
        fi
        
        if [ "$updated" = true ]; then
            echo -e "${GREEN}Added amd64 architecture settings to .env file.${NC}"
        fi
    fi
}

# Run the architecture check
check_architecture_settings

# Function to display help
show_help() {
    echo -e "${GREEN}Frontend Site Development Helper${NC}"
    echo "Usage: ./dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  up       - Start the development environment"
    echo "  down     - Stop the development environment"
    echo "  restart  - Restart the development environment"
    echo "  rebuild  - Rebuild containers and start"
    echo "  logs     - Show logs from all containers"
    echo "  logs-ai  - Show logs from the AI server container"
    echo "  logs-fe  - Show logs from the frontend container"
    echo "  logs-db  - Show logs from the database container"
    echo "  ps       - Show container status"
    echo "  exec     - Open a shell in the frontend container"
    echo "  exec-ai  - Open a shell in the AI server container"
    echo "  db       - Open PostgreSQL CLI in the database container"
    echo "  help     - Show this help message"
}

# Check command argument
case "$1" in
    up)
        echo -e "${GREEN}Starting development environment...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml up -d
        echo -e "${GREEN}Development environment started!${NC}"
        echo "Frontend available at: http://localhost:${PORT}"
        echo "AI Server available at: http://localhost:5002"
        echo "PostgreSQL available at: localhost:5432"
        ;;
    down)
        echo -e "${GREEN}Stopping development environment...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml down
        ;;
    restart)
        echo -e "${GREEN}Restarting development environment...${NC}"
        echo -e "${YELLOW}Stopping containers...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml down
        echo -e "${YELLOW}Starting containers...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml up -d
        echo -e "${GREEN}Development environment restarted!${NC}"
        echo "Frontend available at: http://localhost:${PORT}"
        echo "AI Server available at: http://localhost:5002"
        echo "PostgreSQL available at: localhost:5432"
        ;;
    rebuild)
        echo -e "${GREEN}Rebuilding and starting development environment...${NC}"
        # Set architecture to amd64
        export TARGETARCH=amd64
        export BUILDPLATFORM=linux/amd64
        export TARGETPLATFORM=linux/amd64
        # Build with platform arguments
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml build --build-arg TARGETPLATFORM=linux/amd64 --build-arg BUILDPLATFORM=linux/amd64
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml up -d
        ;;
    logs)
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml logs -f
        ;;
    logs-ai)
        echo -e "${GREEN}Showing logs from AI server container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml logs -f ai_server
        ;;
    logs-fe)
        echo -e "${GREEN}Showing logs from frontend container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml logs -f frontend
        ;;
    logs-db)
        echo -e "${GREEN}Showing logs from database container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml logs -f db
        ;;
    ps)
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml ps
        ;;
    exec)
        echo -e "${GREEN}Opening shell in frontend container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml exec frontend /bin/sh
        ;;
    exec-ai)
        echo -e "${GREEN}Opening shell in AI server container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml exec ai_server /bin/sh
        ;;
    db)
        echo -e "${GREEN}Opening PostgreSQL CLI in database container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml exec db psql -U ${POSTGRES_USER:-db} -d ${POSTGRES_DB:-frontend}
        ;;
    help|*)
        show_help
        ;;
esac 