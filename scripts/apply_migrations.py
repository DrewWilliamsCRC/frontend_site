#!/usr/bin/env python3
"""
Database Migration Script for Production

This script safely applies database migrations to a production database.
It checks for pending migrations and applies them in a safe manner.
"""
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set")
    sys.exit(1)

def get_db_connection():
    """Establish a database connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def ensure_migrations_table():
    """Ensure the migrations tracking table exists."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if migrations table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'migrations'
                );
            """)
            if not cur.fetchone()['exists']:
                # Create migrations table
                cur.execute("""
                    CREATE TABLE migrations (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                print("Created migrations tracking table")
            else:
                print("Migrations table already exists")
    except Exception as e:
        print(f"Error ensuring migrations table: {e}")
        sys.exit(1)
    finally:
        conn.close()

def has_migration_been_applied(migration_name):
    """Check if a specific migration has been applied."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM migrations WHERE migration_name = %s);", 
                      (migration_name,))
            return cur.fetchone()['exists']
    finally:
        conn.close()

def mark_migration_applied(migration_name):
    """Mark a migration as applied in the tracking table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO migrations (migration_name) VALUES (%s);", 
                      (migration_name,))
            conn.commit()
            print(f"Marked migration '{migration_name}' as applied")
    except Exception as e:
        conn.rollback()
        print(f"Error marking migration as applied: {e}")
    finally:
        conn.close()

def ensure_admin_user():
    """Ensure the admin user exists, but don't update if it already exists."""
    if has_migration_been_applied('ensure_admin_user'):
        print("Admin user migration already applied, skipping")
        return
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if admin exists
            cur.execute("SELECT username FROM users WHERE username = 'admin'")
            if cur.fetchone() is None:
                # Create admin user with Werkzeug 3.1.3 compatible hash
                password_hash = generate_password_hash('admin123')
                cur.execute("""
                    INSERT INTO users (
                        username, 
                        email, 
                        password_hash, 
                        news_categories, 
                        city_name
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    'admin',
                    'admin@localhost',
                    password_hash,
                    'general,technology,business',
                    'San Francisco'
                ))
                conn.commit()
                print("Created admin user")
            else:
                print("Admin user already exists, not modifying")
        
        # Mark migration as applied
        mark_migration_applied('ensure_admin_user')
    except Exception as e:
        conn.rollback()
        print(f"Error ensuring admin user: {e}")
    finally:
        conn.close()

def ensure_user_columns():
    """Ensure all required columns exist in the users table."""
    if has_migration_been_applied('ensure_user_columns'):
        print("User columns migration already applied, skipping")
        return
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check columns
            for column, type_info in [
                ('city_name', 'VARCHAR(255)'),
                ('button_width', 'INTEGER DEFAULT 200'),
                ('button_height', 'INTEGER DEFAULT 200')
            ]:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'users' AND column_name = %s
                    );
                """, (column,))
                
                if not cur.fetchone()['exists']:
                    cur.execute(f"ALTER TABLE users ADD COLUMN {column} {type_info};")
                    conn.commit()
                    print(f"Added {column} column to users table")
                else:
                    print(f"Column {column} already exists")
        
        # Mark migration as applied
        mark_migration_applied('ensure_user_columns')
    except Exception as e:
        conn.rollback()
        print(f"Error ensuring user columns: {e}")
    finally:
        conn.close()

def main():
    """Main migration function."""
    print("Starting database migrations...")
    
    # Ensure migrations table exists
    ensure_migrations_table()
    
    # Apply all migrations
    ensure_admin_user()
    ensure_user_columns()
    
    print("Database migrations completed successfully")

if __name__ == "__main__":
    main() 