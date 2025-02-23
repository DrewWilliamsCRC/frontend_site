#!/bin/sh
set -e

# Extract database connection details from DATABASE_URL
DB_HOST="db"
DB_PORT="5432"

# Create log directory
mkdir -p /app/logs
touch /app/logs/gunicorn.log
touch /app/logs/access.log

echo "Starting entrypoint script..."
echo "Database URL: $DATABASE_URL"
echo "Database Host: $DB_HOST"
echo "Database Port: $DB_PORT"
echo "Database User: $POSTGRES_USER"
echo "Database Name: $POSTGRES_DB"
echo "Flask Environment: $FLASK_ENV"
echo "Flask Debug: $FLASK_DEBUG"
echo "Port: $PORT"

# Wait for database to be ready
echo "Waiting for database at ${DB_HOST}:${DB_PORT}..."
timeout=30
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -p "$DB_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\l' 2>/dev/null; do
  timeout=$((timeout - 1))
  if [ $timeout -eq 0 ]; then
    echo "Could not connect to database, timeout"
    exit 1
  fi
  echo "Database not ready, waiting... (${timeout} attempts left)"
  sleep 1
done

echo "Database is ready!"
echo "Testing database connection with full connection string..."
PGPASSWORD=$POSTGRES_PASSWORD psql "$DATABASE_URL" -c '\conninfo'

# Start the Flask application with gunicorn
echo "Starting Gunicorn with 4 workers..."
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

echo "Python version:"
python --version

echo "Checking app.py exists:"
if [ -f "app.py" ]; then
    echo "app.py found"
    echo "Content of first few lines of app.py:"
    head -n 5 app.py
else
    echo "app.py not found!"
    exit 1
fi

echo "Starting Gunicorn..."
exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 4 \
    --worker-class sync \
    --timeout 120 \
    --log-level info \
    --access-logfile /app/logs/access.log \
    --error-logfile /app/logs/gunicorn.log \
    --capture-output \
    --preload \
    --worker-tmp-dir /dev/shm \
    app:app