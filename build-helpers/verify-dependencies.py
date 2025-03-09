#!/usr/bin/env python3
"""
Verify All Required Dependencies

This script verifies that all required Python packages for the application
are properly installed and reports their versions.
"""

import sys
import importlib
import pkg_resources

print(f"Python version: {sys.version}")

# List of all required packages
required_packages = [
    "flask",
    "requests",
    "pandas",
    "numpy",
    "beautifulsoup4",
    "bs4",
    "nltk",
    "praw",
    "scikit-learn",
    "pillow",
    "lxml",
]

# Check each package
print("\nVerifying package installations:")
all_ok = True

for package in required_packages:
    try:
        # Try different import names for some packages
        if package == "beautifulsoup4":
            import_name = "bs4"
        elif package == "pillow":
            import_name = "PIL"
        elif package == "scikit-learn":
            import_name = "sklearn"
        else:
            import_name = package
        
        # Try to import the package
        module = importlib.import_module(import_name)
        
        # Get the version
        try:
            version = pkg_resources.get_distribution(package).version
        except:
            version = getattr(module, "__version__", "unknown")
        
        print(f"✓ {package:<20} - {version}")
    except ImportError as e:
        print(f"✗ {package:<20} - MISSING! {e}")
        all_ok = False

# NLTK specific checks
try:
    import nltk
    required_nltk_data = ["vader_lexicon", "punkt", "stopwords", "wordnet"]
    print("\nVerifying NLTK data:")
    for data in required_nltk_data:
        try:
            nltk.data.find(f"tokenizers/{data}" if data == "punkt" else f"corpora/{data}")
            print(f"✓ NLTK {data:<15} - Installed")
        except LookupError:
            print(f"✗ NLTK {data:<15} - MISSING!")
            all_ok = False
except ImportError:
    print("\nCannot verify NLTK data as NLTK is not installed")

# Final status
print("\nDependency verification result:", "SUCCESS" if all_ok else "FAILURE")
if not all_ok:
    print("Some dependencies are missing - please install them before running the application")
    sys.exit(1)
else:
    print("All required dependencies are installed successfully!") 