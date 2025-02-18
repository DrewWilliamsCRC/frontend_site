#!/usr/bin/env python3
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import getpass
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
from psycopg2 import IntegrityError

# Load environment variables from .env (if used)
load_dotenv()

if os.environ.get("FLASK_ENV") == "development":
    DATABASE_URL = os.environ.get("DEV_DATABASE_URL", os.environ.get("DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set for the Flask application.")

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def create_table(conn):
    """
    Create the users table if it does not exist.
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
    print("Table 'users' ensured exists.")

def list_users(conn):
    """
    List all users in the database.
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
    Add a new user to the database through a CLI prompt.
    """
    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
    password = getpass.getpass("Enter password: ").strip()
    if not password:
        print("Password cannot be empty.")
        return
    # Hash the password (for example purposes, using werkzeug's generate_password_hash)
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
    Delete a user by their ID.
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
    Display the help message listing available commands.
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
    print("Database initialized or updated.")

def main():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        create_table(conn)
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Error: Command required.")
        show_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "init":
        init_db()
    elif command == "list":
        list_users(conn)
    elif command == "add":
        add_user(conn)
    elif command == "delete":
        delete_user(conn)
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()
    
    conn.close()

if __name__ == "__main__":
    main() 
