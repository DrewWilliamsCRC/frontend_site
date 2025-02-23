import os
import sys
from app import init_db

def confirm_production_init():
    """Confirm before initializing production database."""
    env = os.environ.get("FLASK_ENV", "production")
    if env == "production":
        response = input("""
WARNING: You are about to initialize/modify the PRODUCTION database!
This operation:
- Is safe for existing data (uses IF NOT EXISTS)
- Will create missing tables
- Will add new columns if needed
        
Do you want to proceed? [y/N]: """).lower().strip()
        
        return response == 'y'
    return True

if __name__ == "__main__":
    try:
        if confirm_production_init():
            print(f"Initializing database in {os.environ.get('FLASK_ENV', 'production')} mode...")
            init_db()
            print("Database initialization complete.")
        else:
            print("Database initialization cancelled.")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1) 