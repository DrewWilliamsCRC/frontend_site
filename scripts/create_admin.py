#!/usr/bin/env python3

import os
import psycopg2
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("DEV_DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set in environment")

def create_admin_user():
    """Create an admin user if one doesn't exist."""
    # Admin credentials
    username = 'admin'
    password = 'admin123'
    email = 'admin@localhost'
    
    # Generate password hash using Werkzeug
    password_hash = generate_password_hash(password)
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Check if admin exists
        cur.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cur.fetchone() is None:
            # Create admin user
            cur.execute("""
                INSERT INTO users (username, email, password_hash, news_categories)
                VALUES (%s, %s, %s, %s)
            """, (username, email, password_hash, 'general,technology,business'))
            conn.commit()
            print("Admin user created successfully")
        else:
            # Update admin password
            cur.execute("""
                UPDATE users 
                SET password_hash = %s,
                    email = %s
                WHERE username = %s
            """, (password_hash, email, username))
            conn.commit()
            print("Admin user password updated")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    create_admin_user() 