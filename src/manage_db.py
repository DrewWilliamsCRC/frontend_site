#!/usr/bin/env python3

# Database Management Module
# ------------------------
# This module provides a comprehensive set of tools for managing the application's
# database, including user management and custom services. It supports both
# command-line interface (CLI) operations and programmatic access.

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import getpass
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
from psycopg2 import IntegrityError
import click
from flask.cli import with_appcontext

# Load environment variables from .env file
load_dotenv()

# Database Configuration
# --------------------
# Select the appropriate database URL based on the environment
if os.environ.get("FLASK_ENV") == "development":
    # When running in Docker, we should use the "db" service name, not localhost
    DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("DEV_DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set for the Flask application.")

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.extensions.connection: Database connection with RealDictCursor
        
    Raises:
        SystemExit: If connection fails, exits with error message
        
    Note:
        Uses RealDictCursor to return results as dictionaries instead of tuples
    """
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def create_table(conn):
    """
    Creates the required database tables if they don't exist.
    
    Creates two tables:
    1. users:
       - id: Auto-incrementing primary key
       - username: Unique user identifier
       - password_hash: Securely hashed password
       - city_name: User's preferred city
       
    2. custom_services:
       - id: Auto-incrementing primary key
       - user_id: Foreign key to users table
       - name: Service name
       - url: Service URL
       - icon: Icon identifier
       - description: Optional service description
       - section: Service category
       - created_at: Timestamp of creation
       
    Args:
        conn: Active database connection
        
    Note:
        Uses CASCADE on delete for custom_services to automatically
        remove a user's services when the user is deleted
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                city_name TEXT
            );
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS custom_services (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                icon TEXT NOT NULL,
                description TEXT,
                section TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    print("Table 'users' and 'custom_services' ensured exists.")

def list_users(conn):
    """
    Retrieves and displays all users from the database.
    
    Args:
        conn: Active database connection
        
    Output:
        Prints a formatted list of users with their ID, username, and city
        If no users exist, prints "No users found."
    """
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users ORDER BY id;")
        users = cur.fetchall()
    if not users:
        print("No users found.")
    else:
        print("Users:")
        for user in users:
            print(f"ID: {user['id']}, Username: {user['username']}, City: {user['city_name']}")

def add_user(conn):
    """
    Interactively adds a new user to the database through CLI prompts.
    
    Args:
        conn: Active database connection
        
    Process:
        1. Prompts for username (required)
        2. Securely prompts for password (required)
        3. Prompts for city name (optional)
        4. Hashes the password using Werkzeug's implementation
        5. Stores the user data in the database
        
    Note:
        - Handles duplicate username conflicts
        - Uses secure password input (no echo)
        - Implements proper transaction management
    """
    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
    password = getpass.getpass("Enter password: ").strip()
    if not password:
        print("Password cannot be empty.")
        return
    # Hash the password using Werkzeug's secure implementation
    password_hash = generate_password_hash(password)
    city_name = input("Enter city name (optional): ").strip()
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, city_name) VALUES (%s, %s, %s)",
                (username, password_hash, city_name),
            )
        conn.commit()
        print("User added successfully.")
    except IntegrityError:
        conn.rollback()
        print("Error: Username already exists.")
    except Exception as e:
        conn.rollback()
        print(f"Error adding user: {e}")

def delete_user(conn):
    """
    Deletes a user and their associated data from the database.
    
    Args:
        conn: Active database connection
        
    Process:
        1. Prompts for user ID
        2. Validates ID format
        3. Deletes user if found
        4. Associated custom_services are automatically deleted via CASCADE
        
    Note:
        Implements proper transaction management with rollback on error
    """
    user_id = input("Enter user ID to delete: ").strip()
    if not user_id.isdigit():
        print("Invalid ID.")
        return
    user_id = int(user_id)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
        conn.commit()
        print("User deleted successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error deleting user: {e}")

def show_help():
    """
    Displays help information for the database management commands.
    
    Lists all available commands with their descriptions:
    - init: Database initialization
    - list: User listing
    - add: User addition
    - delete: User deletion
    - help: Help display
    """
    help_text = """
Usage: python manage_db.py [command]

Commands:
    init      - Initialize the database (create tables).
    list      - List all users in the database.
    add       - Add a new user.
    delete    - Delete a user by ID.
    help      - Show this help message.
"""
    print(help_text)

def init_db():
    """
    Initializes the database schema by creating required tables.
    
    This function is similar to create_table() but operates independently
    with its own connection management. It's primarily used during
    application setup and testing.
    
    Note:
        Automatically commits changes and properly closes the connection
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    city_name TEXT
                );
            """)
            conn.commit()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS custom_services (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    icon TEXT NOT NULL,
                    description TEXT,
                    section TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
    print("Database initialized or updated.")

# Click CLI Command Group
# ---------------------
@click.group()
def cli():
    """Database management command group for Click CLI interface."""
    pass

@cli.command()
@with_appcontext
def init():
    """
    Click command to initialize the database tables.
    
    This command is accessible via the Flask CLI and creates
    all necessary database tables using the application's
    database connection.
    """
    from app import get_db_connection
    conn = get_db_connection()
    try:
        create_table(conn)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()

@cli.command()
@with_appcontext
def add_user_cli():
    """
    Click command to add a new user through the CLI.
    
    This command provides a CLI wrapper around the add_user function,
    managing the database connection within the Flask application context.
    """
    from app import get_db_connection
    conn = get_db_connection()
    try:
        add_user(conn)
    finally:
        conn.close()

if __name__ == '__main__':
    # Add the parent directory to sys.path to allow importing app module
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    cli() 
