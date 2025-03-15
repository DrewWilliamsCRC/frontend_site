#!/usr/bin/env python
"""
Script to test the admin password hash.
"""
from werkzeug.security import generate_password_hash, check_password_hash
import sys

def main():
    # Admin password 
    password = "admin123"
    
    # The hash we're now using in our SQL script (Werkzeug 3.1.3 compatible)
    werkzeug_3_hash = 'scrypt:32768:8:1$BymU2FAmwmRqVLMp$3257820b2e6dfcfe5c71a02a05e0f52a4eb51797990de4a6b4a64a661bac21aad0d6c0991e79cc8571c41486a5d77e142484321ece91ec449a21c9ace0f59fdb'
    
    # The previously tried hash for Werkzeug 2.0.0
    werkzeug_2_hash = 'pbkdf2:sha256:150000$ImgIqXwD$ef4439eb8a6bf8d211997af5d100b80a768f3153c06e696335076b4918019150'
    
    # The originally used hash that wasn't working
    original_hash = 'pbkdf2:sha256:260000$OgxHQQrntlpZuKxf$90aea45be3fd9b1de2323d26cf53f4b05eb8d91a5b95bab7e69cc492d76b6c48'
    
    # Check validation with all hashes
    werkzeug_3_valid = check_password_hash(werkzeug_3_hash, password)
    werkzeug_2_valid = check_password_hash(werkzeug_2_hash, password)
    original_valid = check_password_hash(original_hash, password)
    
    print(f"Password: {password}")
    print(f"Werkzeug 3.1.3 hash validates: {werkzeug_3_valid}")
    print(f"Werkzeug 2.0.0 hash validates: {werkzeug_2_valid}")
    print(f"Original hash validates: {original_valid}")
    
    # Generate a fresh hash to show the format
    fresh_hash = generate_password_hash(password)
    print(f"\nFreshly generated hash: {fresh_hash}")
    print(f"Fresh hash validates: {check_password_hash(fresh_hash, password)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 