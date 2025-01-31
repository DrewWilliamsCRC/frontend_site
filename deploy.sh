#!/usr/bin/env bash
set -e

# Sync the code from the frontend_site to the docker/frontend directory
rsync -av /docker/code-server/projects/frontend_site/ /docker/frontend/

# Bring down existing containers, remove orphans, volumes, and images
docker compose down --remove-orphans --volumes --rmi all

# Bring the services back up in detached mode
docker compose up -d