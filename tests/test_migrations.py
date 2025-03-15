#!/usr/bin/env python3
"""
Test suite for the database migration system.

These tests verify that:
1. The migrations table is properly created
2. Migrations are properly tracked and not reapplied
3. The admin user is created correctly
4. Required columns are added if missing
"""
import unittest
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash

# Add the parent directory to the path so we can import the migration script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.apply_migrations import (
    get_db_connection, ensure_migrations_table, 
    has_migration_been_applied, mark_migration_applied,
    ensure_admin_user, ensure_user_columns
)

class TestMigrations(unittest.TestCase):
    """Test the database migration system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test database."""
        # Get database connection details from environment variables
        cls.database_url = os.environ.get("DATABASE_URL")
        if not cls.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Connect to the database
        cls.conn = psycopg2.connect(cls.database_url, cursor_factory=RealDictCursor)
        
        # Create a clean slate for testing
        cls._clean_database()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test database."""
        cls._clean_database()
        cls.conn.close()
    
    @classmethod
    def _clean_database(cls):
        """Remove test data and tables."""
        with cls.conn.cursor() as cur:
            # Drop the migrations table if it exists
            cur.execute("DROP TABLE IF EXISTS migrations;")
            
            # Remove the admin user from the users table if it exists
            cur.execute("DELETE FROM users WHERE username = 'admin';")
            
            # Commit the changes
            cls.conn.commit()
    
    def test_01_migrations_table_creation(self):
        """Test that the migrations table is properly created."""
        # Ensure the migrations table doesn't exist
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS migrations;")
            self.conn.commit()
        
        # Create the migrations table
        ensure_migrations_table()
        
        # Check that the migrations table exists
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'migrations'
                );
            """)
            self.assertTrue(cur.fetchone()['exists'])
    
    def test_02_migration_tracking(self):
        """Test that migrations are properly tracked."""
        # Ensure the migrations table exists
        ensure_migrations_table()
        
        # Check if a non-existent migration has been applied
        self.assertFalse(has_migration_been_applied('test_migration'))
        
        # Mark the migration as applied
        mark_migration_applied('test_migration')
        
        # Check if the migration has been applied now
        self.assertTrue(has_migration_been_applied('test_migration'))
    
    def test_03_admin_user_creation(self):
        """Test that the admin user is created correctly."""
        # Ensure the migrations table exists
        ensure_migrations_table()
        
        # Delete any existing admin user
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = 'admin';")
            self.conn.commit()
        
        # Reset the migration tracking
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM migrations WHERE migration_name = 'ensure_admin_user';")
            cur.execute("DELETE FROM migrations WHERE migration_name = 'ensure_user_columns';")
            self.conn.commit()
        
        # First ensure required columns exist before creating admin user
        ensure_user_columns()
        
        # Then create the admin user
        ensure_admin_user()
        
        # Check if the admin user exists
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = 'admin';")
            admin = cur.fetchone()
            
            # Verify the admin user details
            self.assertIsNotNone(admin)
            self.assertEqual(admin['username'], 'admin')
            self.assertEqual(admin['email'], 'admin@localhost')
            self.assertTrue(check_password_hash(admin['password_hash'], 'admin123'))
        
        # Check that the migration was marked as applied
        self.assertTrue(has_migration_been_applied('ensure_admin_user'))
        
        # Call ensure_admin_user again and check it doesn't create a duplicate
        # First, count the users
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users WHERE username = 'admin';")
            initial_count = cur.fetchone()['count']
        
        # Call the function again
        ensure_admin_user()
        
        # Check the count hasn't changed
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users WHERE username = 'admin';")
            new_count = cur.fetchone()['count']
        
        self.assertEqual(initial_count, new_count)
    
    def test_04_user_columns(self):
        """Test that required columns are added if missing."""
        # Ensure the migrations table exists
        ensure_migrations_table()
        
        # Reset the migration tracking
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM migrations WHERE migration_name = 'ensure_user_columns';")
            self.conn.commit()
        
        # Ensure columns exist
        ensure_user_columns()
        
        # Check that the columns exist
        with self.conn.cursor() as cur:
            for column in ['city_name', 'button_width', 'button_height']:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'users' AND column_name = %s
                    );
                """, (column,))
                self.assertTrue(cur.fetchone()['exists'], f"Column {column} does not exist")
        
        # Check that the migration was marked as applied
        self.assertTrue(has_migration_been_applied('ensure_user_columns'))

if __name__ == '__main__':
    unittest.main() 