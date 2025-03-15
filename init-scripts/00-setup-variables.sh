#!/bin/bash
set -e

# This script is used to set up initialization information
echo "Initializing database..."
echo "POSTGRES_DB: $POSTGRES_DB"
echo "POSTGRES_USER: $POSTGRES_USER"

# Detect environment
if [ "$FLASK_ENV" = "production" ]; then
    echo "Running in PRODUCTION environment"
    echo "In production, database will be preserved between restarts"
    echo "Admin user password will only be set on first initialization"
else
    echo "Running in DEVELOPMENT environment"
    echo "Admin email will be set to: ${ADMIN_EMAIL:-admin@localhost}"
    echo "Admin password will be set to: admin123 (please change after first login)"
fi

# No need to generate password hash here anymore - it's hardcoded in the SQL file
echo "Database initialization started successfully" 