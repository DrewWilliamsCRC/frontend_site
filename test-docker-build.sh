#!/bin/bash

# Exit on any error
set -e

echo "🔨 Building Docker container..."
docker build -t frontend-site-test .

echo "🚀 Starting container..."
docker run -d -p 5001:5001 --name frontend-site-test frontend-site-test

echo "⏳ Waiting for container to start..."
sleep 3

echo "🔍 Checking container status..."
if docker ps | grep -q frontend-site-test; then
    echo "✅ Container is running successfully!"
else
    echo "❌ Container failed to start"
    exit 1
fi

echo "🧹 Cleaning up..."
docker stop frontend-site-test
docker rm frontend-site-test

echo "✨ Test completed successfully!"
