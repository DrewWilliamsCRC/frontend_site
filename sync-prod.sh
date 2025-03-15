#!/bin/bash

# Configuration
PROD_USER="drew"          # Replace with your server username
PROD_DIR="/docker/frontend"     # Replace with your deployment directory
SSH_CONFIG="production-server"   # Use the Host name from SSH config

# SSH Control settings
SOCKET_DIR="/tmp/${USER}_ssh_sockets"
SOCKET_FILE="${SOCKET_DIR}/frontend_deploy"
mkdir -p "${SOCKET_DIR}"
chmod 700 "${SOCKET_DIR}"

# Essential files and directories to sync
ESSENTIAL_FILES=(
    "docker-compose.prod.yml"
    "Dockerfile.ai"
    "dockerfile"
    "docker-entrypoint.sh"
    "ai_entrypoint.sh"
    "app.py"
    "ai_server.py"
    "requirements.txt"
    "requirements-frontend.txt"
    "wsgi.py"
    "update_containers.sh"
)

ESSENTIAL_DIRS=(
    "build-helpers"
    "init-scripts"
    "templates"
    "static"
    "src"
    "ai_experiments"
)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up SSH connection...${NC}"
    if [ -S "${SOCKET_FILE}" ]; then
        ssh -S "${SOCKET_FILE}" -O exit ${SSH_CONFIG} 2>/dev/null
        rm -f "${SOCKET_FILE}"
    fi
}

# Error handler
error_handler() {
    echo -e "\n${RED}Error occurred. Cleaning up...${NC}"
    cleanup
    exit 1
}

# Set up cleanup on script exit and error
trap cleanup EXIT
trap error_handler ERR

# Validation function for critical files
validate_critical_files() {
    local missing_files=()
    local critical_files=(
        "Dockerfile.ai"
        "docker-compose.prod.yml"
        "ai_entrypoint.sh"
        "ai_server.py"
        ".env.prod"  # Check for .env.prod instead of .env
        "build-helpers/ai-critical-requirements.txt"
        "ai_experiments/alpha_vantage_pipeline.py"
    )

    for file in "${critical_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing critical files:${NC}"
        printf '%s\n' "${missing_files[@]}"
        exit 1
    fi
}

# Validation function for environment variables
validate_env_file() {
    if [ ! -f ".env.prod" ]; then
        echo -e "${RED}Error: .env.prod file not found${NC}"
        exit 1
    fi

    local required_vars=(
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "POSTGRES_DB"
        "ALPHA_VANTAGE_API_KEY"
    )

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env.prod; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required environment variables in .env.prod:${NC}"
        printf '%s\n' "${missing_vars[@]}"
        exit 1
    fi
}

# Validation function for build-helpers directory
validate_build_helpers() {
    if [ ! -d "build-helpers" ]; then
        echo -e "${RED}Error: build-helpers directory not found${NC}"
        exit 1
    fi

    local required_files=(
        "ai-critical-requirements.txt"
        "install-deps.sh"
    )

    local missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "build-helpers/$file" ]; then
            missing_files+=("build-helpers/$file")
        fi
    done

    if [ ${#missing_files[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required files in build-helpers:${NC}"
        printf '%s\n' "${missing_files[@]}"
        exit 1
    fi
}

echo -e "${YELLOW}Starting sync to production server...${NC}"

# Run validation checks
echo -e "${GREEN}Validating deployment files...${NC}"
validate_critical_files
validate_env_file
validate_build_helpers
echo -e "${GREEN}Validation successful${NC}"

# Establish master SSH connection with control socket
echo -e "${GREEN}Establishing SSH connection...${NC}"
echo -e "${YELLOW}Using SSH key from config file...${NC}"

# Create the master connection
ssh -M -S "${SOCKET_FILE}" -o "ControlPersist=yes" -t ${SSH_CONFIG} "echo 'SSH connection successful'" || {
    echo -e "${RED}Failed to establish SSH connection${NC}"
    exit 1
}

echo -e "${GREEN}SSH connection established successfully${NC}"

# Function to run SSH commands using the control socket
run_ssh() {
    ssh -S "${SOCKET_FILE}" ${SSH_CONFIG} "$@"
}

# Create necessary directories
echo -e "${GREEN}Creating directories on production server...${NC}"
for dir in "${ESSENTIAL_DIRS[@]}"; do
    echo "Creating ${dir}..."
    run_ssh "mkdir -p ${PROD_DIR}/${dir}"
done

# Sync files using the established connection
echo -e "${GREEN}Syncing files to production server...${NC}"
for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Syncing $file..."
        rsync -e "ssh -S ${SOCKET_FILE}" -avz --progress "$file" ${SSH_CONFIG}:${PROD_DIR}/
    else
        echo -e "${YELLOW}Warning: $file not found${NC}"
    fi
done

# Copy .env.prod to the server as .env
echo -e "${GREEN}Copying .env.prod to server as .env...${NC}"
if [ -f ".env.prod" ]; then
    rsync -e "ssh -S ${SOCKET_FILE}" -avz --progress ".env.prod" ${SSH_CONFIG}:${PROD_DIR}/"/.env"
else
    echo -e "${RED}Error: .env.prod file not found${NC}"
    exit 1
fi

# Sync directories using the established connection
echo -e "${GREEN}Syncing directories to production server...${NC}"
for dir in "${ESSENTIAL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Syncing $dir/..."
        rsync -e "ssh -S ${SOCKET_FILE}" -avz --progress "$dir/" ${SSH_CONFIG}:${PROD_DIR}/${dir}/
    else
        echo -e "${YELLOW}Warning: $dir not found${NC}"
    fi
done

echo -e "${GREEN}Sync completed!${NC}"

# Start services one by one and check logs
echo -e "${GREEN}Starting services one by one...${NC}"

# Check Docker Compose version on the server
echo -e "${YELLOW}Checking Docker Compose command format...${NC}"
if run_ssh "command -v docker-compose >/dev/null 2>&1"; then
    DOCKER_COMPOSE_CMD="docker-compose -f docker-compose.prod.yml"
    echo -e "${GREEN}Using docker-compose command (v1 format)${NC}"
else
    DOCKER_COMPOSE_CMD="docker compose -f docker-compose.prod.yml"
    echo -e "${GREEN}Using docker compose command (v2 format)${NC}"
fi

# Start and check database first
echo -e "${YELLOW}Starting database...${NC}"

# Clean up and recreate postgres volume
echo -e "${YELLOW}Cleaning up postgres volume...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} down" || {
    echo -e "${RED}Failed to stop services. Trying alternative approach...${NC}"
    run_ssh "cd ${PROD_DIR} && docker stop \$(docker ps -a -q) || true"
}

# Remove volumes with error handling
echo -e "${YELLOW}Removing old volumes...${NC}"
run_ssh "cd ${PROD_DIR} && docker volume rm frontend_postgres_data frontend_ai_data frontend_ai_models frontend_ai_cache || true"

# Create volumes
echo -e "${YELLOW}Creating new volumes...${NC}"
run_ssh "cd ${PROD_DIR} && docker volume create frontend_postgres_data || true"
run_ssh "cd ${PROD_DIR} && docker volume create frontend_ai_data || true"
run_ssh "cd ${PROD_DIR} && docker volume create frontend_ai_models || true"
run_ssh "cd ${PROD_DIR} && docker volume create frontend_ai_cache || true"

# Initialize AI volumes with correct permissions
echo -e "${YELLOW}Initializing AI volumes with correct permissions...${NC}"
run_ssh "cd ${PROD_DIR} && docker run --rm \
    -v frontend_ai_data:/data \
    -v frontend_ai_models:/models \
    -v frontend_ai_cache:/cache \
    alpine sh -c 'mkdir -p /data/sectors /data/ml_ready/indices && \
                 chown -R 1000:1000 /data /models /cache && \
                 chmod -R 755 /data /models /cache'" || {
    echo -e "${YELLOW}Volume initialization failed, but continuing...${NC}"
}

# Check volume permissions
echo -e "${YELLOW}Checking AI volume permissions...${NC}"
run_ssh "cd ${PROD_DIR} && docker run --rm \
    -v frontend_ai_data:/data \
    -v frontend_ai_models:/models \
    -v frontend_ai_cache:/cache \
    alpine sh -c 'ls -la /data /models /cache'" || {
    echo -e "${YELLOW}Volume permission check failed, but continuing...${NC}"
}

# Check postgres data volume permissions
echo -e "${YELLOW}Checking postgres data volume...${NC}"
run_ssh "cd ${PROD_DIR} && docker volume inspect frontend_postgres_data" || {
    echo -e "${YELLOW}Volume inspection failed, but continuing...${NC}"
}

# Start database
echo -e "${YELLOW}Starting database service...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} up -d db" || {
    echo -e "${RED}Failed to start database. Please check server logs.${NC}"
    exit 1
}

echo -e "${YELLOW}Waiting 15 seconds for database to initialize...${NC}"
run_ssh "sleep 15"

echo -e "${YELLOW}Checking database logs...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} logs db" || {
    echo -e "${YELLOW}Failed to get logs, but continuing...${NC}"
}

echo -e "${YELLOW}Checking database container status...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} ps db" || {
    echo -e "${YELLOW}Failed to check status, but continuing...${NC}"
}

# Start and check AI server
echo -e "${YELLOW}Starting AI server...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} up -d ai_server" || {
    echo -e "${RED}Failed to start AI server. Please check server logs.${NC}"
    exit 1
}

echo -e "${YELLOW}Checking AI server logs...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} logs ai_server" || {
    echo -e "${YELLOW}Failed to get logs, but continuing...${NC}"
}

# Start and check frontend
echo -e "${YELLOW}Starting frontend...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} up -d frontend" || {
    echo -e "${RED}Failed to start frontend. Please check server logs.${NC}"
    exit 1
}

echo -e "${YELLOW}Checking frontend logs...${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} logs frontend" || {
    echo -e "${YELLOW}Failed to get logs, but continuing...${NC}"
}

# Show overall status
echo -e "${GREEN}Showing overall status:${NC}"
run_ssh "cd ${PROD_DIR} && ${DOCKER_COMPOSE_CMD} ps" || {
    echo -e "${YELLOW}Failed to show status, but deployment should be complete.${NC}"
}

echo -e "${GREEN}Deployment complete!${NC}" 