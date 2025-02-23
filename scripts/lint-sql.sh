#!/bin/bash

# Exit on error
set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if sqlfluff is installed
if ! command -v sqlfluff &> /dev/null; then
    echo "SQLFluff not found. Installing development requirements..."
    pip install -r requirements-dev.txt
fi

# Run SQLFluff on all SQL files
echo "Running SQLFluff on SQL files..."
find . -type f -name "*.sql" -not -path "./venv/*" -print0 | while IFS= read -r -d '' file; do
    echo "Linting $file..."
    sqlfluff lint "$file"
done

# Format SQL files if --fix flag is provided
if [[ "$1" == "--fix" ]]; then
    echo "Fixing SQL files..."
    find . -type f -name "*.sql" -not -path "./venv/*" -print0 | while IFS= read -r -d '' file; do
        echo "Fixing $file..."
        sqlfluff fix "$file" --force
    done
fi

echo "SQL linting complete!" 