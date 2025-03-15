#!/bin/bash

# Rebuild script to ensure matplotlib and other scientific packages are installed

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Explicitly set architecture to amd64
export TARGETARCH=amd64
export BUILDPLATFORM=linux/amd64
export TARGETPLATFORM=linux/amd64

echo -e "${YELLOW}Stopping existing containers...${NC}"
./dev.sh down

echo -e "${YELLOW}Removing old images to force rebuild...${NC}"
docker image rm frontend:test 2>/dev/null || true

echo -e "${GREEN}Building and starting containers with scientific packages...${NC}"
./dev.sh rebuild

echo -e "${YELLOW}Verifying package installation...${NC}"
docker compose exec frontend pip list | grep -E "pandas|numpy|matplotlib|seaborn|scikit-learn"

echo -e "${GREEN}Testing AI Insights endpoint...${NC}"
curl -s http://localhost:${PORT}/api/ai-insights | grep -v "error" && echo -e "${GREEN}Success! AI Insights is working correctly.${NC}" || echo -e "${RED}Error: AI Insights still has issues.${NC}"

echo -e "${GREEN}Done! You can now access the AI Insights page at http://localhost:${PORT}/ai-insights${NC}" 