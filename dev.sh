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
    echo "  logs     - Show logs from containers"
    echo "  ps       - Show container status"
    echo "  exec     - Open a shell in the frontend container"
    echo "  db       - Open PostgreSQL CLI in the database container"
    echo "  help     - Show this help message"
}

# Check command argument
case "$1" in
    up)
        echo -e "${GREEN}Starting development environment...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml up -d
        echo -e "${GREEN}Development environment started!${NC}"
        echo "Frontend available at: http://localhost:5001"
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
        echo "Frontend available at: http://localhost:5001"
        echo "PostgreSQL available at: localhost:5432"
        ;;
    rebuild)
        echo -e "${GREEN}Rebuilding and starting development environment...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml up -d --build
        ;;
    logs)
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml logs -f
        ;;
    ps)
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml ps
        ;;
    exec)
        echo -e "${GREEN}Opening shell in frontend container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml exec frontend /bin/bash
        ;;
    db)
        echo -e "${GREEN}Opening PostgreSQL CLI in database container...${NC}"
        ${DOCKER_COMPOSE} -f docker-compose.dev.yml exec db psql -U ${POSTGRES_USER:-db} -d ${POSTGRES_DB:-frontend_db}
        ;;
    help|*)
        show_help
        ;;
esac 