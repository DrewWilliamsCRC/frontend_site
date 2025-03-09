#!/bin/sh
set -e  # Exit on error

# Display Python and pip versions for debugging
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Explicitly install BeautifulSoup regardless of environment
echo "Explicitly installing BeautifulSoup packages..."
pip install --no-cache-dir beautifulsoup4==4.12.3 bs4==0.0.2 lxml==5.1.0 nltk==3.8.1

# Detect Alpine Linux
if [ -f /etc/alpine-release ]; then
    echo "Alpine Linux detected, using Alpine-specific requirements"
    
    # Make sure pip is upgraded first to handle newer packaging formats
    pip install --upgrade pip setuptools wheel

    echo "Installing Alpine requirements from build-helpers/requirements-alpine.txt"
    
    # Install critical scientific packages first
    echo "Installing critical scientific packages first..."
    pip install --no-cache-dir pandas numpy matplotlib seaborn scikit-learn || true
    
    # Installation strategy: 
    # 1. Try batch installation first for most packages (much faster)
    # 2. If that fails, fall back to one-by-one installation for problematic packages
    
    # Create a list of known problematic packages that need special handling
    PROBLEMATIC_PACKAGES="torch prophet tensorflow mlflow gymnasium"
    
    # First, attempt to install all packages at once (except problematic ones)
    echo "Batch installing non-problematic packages..."
    grep -v -E "($PROBLEMATIC_PACKAGES)" build-helpers/requirements-alpine.txt > /tmp/batch_requirements.txt
    pip install --no-cache-dir -r /tmp/batch_requirements.txt || true
    
    # Then handle problematic packages individually
    echo "Installing potentially problematic packages individually..."
    for pkg in $PROBLEMATIC_PACKAGES; do
        # Find the line containing this package if it exists
        pkg_line=$(grep -E "^$pkg(==|>=|~=|$)" build-helpers/requirements-alpine.txt || echo "")
        if [ -n "$pkg_line" ]; then
            echo "Attempting to install: $pkg_line"
            pip install --no-cache-dir $pkg_line || echo "Failed to install $pkg_line, continuing anyway"
        fi
    done
    
    # If dev requirements exist, install them too
    if [ -f requirements-dev.txt ]; then
        echo "Installing development requirements from requirements-dev.txt"
        pip install --no-cache-dir -r requirements-dev.txt || true
    fi
else
    echo "Non-Alpine Linux detected, using standard requirements"
    pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir -r requirements.txt
    
    # If dev requirements exist, install them too
    if [ -f requirements-dev.txt ]; then
        pip install --no-cache-dir -r requirements-dev.txt || true
    fi
fi

# Verify critical packages are installed
echo "Verifying critical packages..."
for pkg in Flask gunicorn psycopg2 pandas numpy matplotlib seaborn scikit-learn bs4; do
    if pip show $pkg > /dev/null 2>&1; then
        echo "✓ $pkg is installed"
    else
        echo "✗ $pkg is NOT installed"
    fi
done

# Run verification script for BeautifulSoup if it exists
if [ -f ./verify-bs4.py ]; then
    echo "Running BeautifulSoup verification script..."
    python ./verify-bs4.py
fi

# Download NLTK data
if [ -f ./download-nltk-data.py ]; then
    echo "Downloading NLTK data packages..."
    python ./download-nltk-data.py
fi

# Verify all dependencies
if [ -f ./verify-dependencies.py ]; then
    echo "Verifying all required dependencies..."
    python ./verify-dependencies.py
fi 