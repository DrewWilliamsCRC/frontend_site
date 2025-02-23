import os
import sys
from app import init_db, get_db_connection

def check_tables():
    """Check which tables exist in the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row['table_name'] for row in cur.fetchall()]
            print("\nExisting tables:", tables)
            
            # Check specifically for api_usage table
            if 'api_usage' not in tables:
                print("\nWARNING: api_usage table is missing!")
            return tables
    except Exception as e:
        print(f"\nError checking tables: {e}")
        return []
    finally:
        conn.close()

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
        print(f"\nCurrent environment: {os.environ.get('FLASK_ENV', 'production')}")
        print(f"Database URL: {os.environ.get('DATABASE_URL', 'not set')}")
        
        # Check existing tables before initialization
        print("\nChecking existing tables...")
        existing_tables = check_tables()
        
        if confirm_production_init():
            print("\nInitializing database...")
            init_db()
            print("\nDatabase initialization complete.")
            
            # Check tables after initialization
            print("\nVerifying tables after initialization...")
            final_tables = check_tables()
            
            # Compare tables
            new_tables = set(final_tables) - set(existing_tables)
            if new_tables:
                print(f"\nNewly created tables: {new_tables}")
            else:
                print("\nNo new tables were created.")
        else:
            print("\nDatabase initialization cancelled.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError initializing database: {e}")
        sys.exit(1) 