#!/usr/bin/env python
"""
Script to generate and test a password hash for the admin user using Werkzeug 3.1.3.
"""
from werkzeug.security import generate_password_hash, check_password_hash
import sys

def main():
    # Admin password
    password = "admin123"
    
    # Generate a hash with Werkzeug 3.1.3
    password_hash = generate_password_hash(password)
    
    # Test if the hash validates
    is_valid = check_password_hash(password_hash, password)
    
    print(f"Password: {password}")
    print(f"Generated hash: {password_hash}")
    print(f"Hash validates: {is_valid}")
    print(f"Hash length: {len(password_hash)}")
    print(f"Hash method: {password_hash.split('$')[0]}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 