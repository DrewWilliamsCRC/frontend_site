#!/usr/bin/env python
"""
Script to generate a password hash for the admin user.
Used during Docker container initialization.
"""
from werkzeug.security import generate_password_hash
import os
import sys

def main():
    # Get the password from environment variable or use default
    password = os.environ.get('ADMIN_PASSWORD', 'admin123')
    
    # Generate the password hash
    password_hash = generate_password_hash(password)
    
    # Print to stdout for capture by shell scripts
    print(password_hash)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 