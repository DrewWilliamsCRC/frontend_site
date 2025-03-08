#!/bin/bash

# Script to fix any architecture-related issues in the project
# This script ensures all builds use amd64 architecture

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Architecture Fix Tool${NC}"
echo "This script will ensure your project uses amd64 architecture consistently."

# Check if running on an M1/M2 Mac
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "${YELLOW}Detected ARM-based Mac (M1/M2).${NC}"
    echo "This script will configure your project to build for amd64 compatibility."
fi

# Update .env file
echo -e "${GREEN}Checking .env file...${NC}"
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating one...${NC}"
    touch .env
fi

# Update or add architecture settings
echo "Setting architecture to amd64 in .env..."
if grep -q "TARGETARCH=" .env; then
    sed -i.bak 's/TARGETARCH=.*/TARGETARCH=amd64/g' .env
else
    echo "TARGETARCH=amd64" >> .env
fi

if grep -q "BUILDPLATFORM=" .env; then
    sed -i.bak 's/BUILDPLATFORM=.*/BUILDPLATFORM=linux\/amd64/g' .env
else
    echo "BUILDPLATFORM=linux/amd64" >> .env
fi

if grep -q "TARGETPLATFORM=" .env; then
    sed -i.bak 's/TARGETPLATFORM=.*/TARGETPLATFORM=linux\/amd64/g' .env
else
    echo "TARGETPLATFORM=linux/amd64" >> .env
fi

echo -e "${GREEN}.env file updated successfully.${NC}"

# Check and fix Docker configuration
echo -e "${GREEN}Checking Docker configuration...${NC}"
if command -v docker &> /dev/null; then
    # Check if Docker daemon has buildx installed
    if ! docker buildx ls &> /dev/null; then
        echo -e "${YELLOW}Docker buildx not found. Setting it up...${NC}"
        docker buildx create --use
    fi
    
    echo "Setting up Docker to handle cross-platform builds correctly..."
    docker buildx inspect default --bootstrap
else
    echo -e "${RED}Docker not found. Please install Docker to continue.${NC}"
    exit 1
fi

# Fix permissions
echo -e "${GREEN}Making scripts executable...${NC}"
chmod +x dev.sh
chmod +x test-docker-build.sh
chmod +x rebuild-with-matplotlib.sh
chmod +x fix-arch.sh

echo -e "${GREEN}Architecture fix completed!${NC}"
echo "Your project is now configured to use amd64 architecture consistently."
echo "To rebuild with the correct architecture, run:"
echo "./rebuild-with-matplotlib.sh" 