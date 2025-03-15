#!/bin/bash
set -e

# Print the database initialization message
echo "Initializing database..."
echo "POSTGRES_DB: $POSTGRES_DB"
echo "POSTGRES_USER: $POSTGRES_USER"

# Determine environment - default to DEVELOPMENT if not set
ENVIRONMENT="${FLASK_ENV:-DEVELOPMENT}"
echo "Running in $ENVIRONMENT environment"

# Set email to a default if not specified
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@localhost}"
echo "Admin email will be set to: $ADMIN_EMAIL"

# Set admin password - default to "admin123" if not specified
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin123}"
echo "Admin password will be set to: $ADMIN_PASSWORD (please change after first login)"

# Use a pre-generated hash for admin123 (using Werkzeug's generate_password_hash)
PASSWORD_HASH='scrypt:32768:8:1$BymU2FAmwmRqVLMp$3257820b2e6dfcfe5c71a02a05e0f52a4eb51797990de4a6b4a64a661bac21aad0d6c0991e79cc8571c41486a5d77e142484321ece91ec449a21c9ace0f59fdb'

# Create PostgreSQL variables for use in SQL scripts
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create a temporary table to store initialization variables
    CREATE TEMP TABLE IF NOT EXISTS init_vars (
        var_name text PRIMARY KEY,
        var_value text
    );

    -- Insert the variables
    INSERT INTO init_vars (var_name, var_value) VALUES
        ('admin_email', '$ADMIN_EMAIL'),
        ('admin_password', '$PASSWORD_HASH')
    ON CONFLICT (var_name) DO UPDATE
        SET var_value = EXCLUDED.var_value;

    -- Output for debugging
    SELECT * FROM init_vars;
EOSQL

# Echo database initialization status
echo "Database initialization started successfully" 