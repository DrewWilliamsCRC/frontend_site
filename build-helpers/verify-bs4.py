#!/usr/bin/env python3
"""
Verify BeautifulSoup Installation
This script verifies that BeautifulSoup is properly installed and reports version information.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Path: {sys.path}")

try:
    import bs4
    print(f"BeautifulSoup4 is installed: {bs4.__version__}")
    from bs4 import BeautifulSoup
    print("Successfully imported BeautifulSoup class")
except ImportError as e:
    print(f"Failed to import bs4: {e}")
    
try:
    import beautifulsoup4
    print(f"beautifulsoup4 package is available")
except ImportError:
    print("beautifulsoup4 package is NOT directly importable (normal behavior)")

# Check installed packages
print("\nInstalled packages:")
import pkg_resources
for package in pkg_resources.working_set:
    if "soup" in package.key.lower():
        print(f"  - {package.key}: {package.version}") 