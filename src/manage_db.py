import sqlite3
import sys
import getpass

from werkzeug.security import generate_password_hash

# Use a consistent database file path (update accordingly if running in production or development)
DB_NAME = '/docker/frontend/data/users.db'  # or adjust to match your environment

def hash_password(password):
    """
    Returns a hashed version of the given password using Werkzeug's generate_password_hash.
    This ensures compatibility with check_password_hash used during login.
    """
    return generate_password_hash(password)

def create_table(conn):
    """
    Ensures the 'users' table exists with the updated schema.
    """
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                city_name TEXT
            );
        ''')

def list_users(conn):
    """
    Lists all users in the database with their ID, username, and city name.
    """
    cursor = conn.execute("SELECT id, username, city_name FROM users")
    rows = cursor.fetchall()
    if rows:
        print("\nUsers:")
        for row in rows:
            print(f"ID: {row[0]}, Username: {row[1]}, City: {row[2]}")
    else:
        print("\nNo users found.")

def add_user(conn):
    """
    Adds a new user to the database with a hashed password.
    """
    username = input("Enter username: ")
    password = getpass.getpass("Enter password: ")
    city_name = input("Enter city name (optional): ")
    password_hash = hash_password(password)
    try:
        with conn:
            conn.execute("INSERT INTO users (username, password_hash, city_name) VALUES (?, ?, ?)", 
                         (username, password_hash, city_name))
        print("User added successfully.")
    except sqlite3.IntegrityError as e:
        print(f"Error adding user: {e}")

def update_user(conn):
    """
    Updates an existing user's information in the database, including hashed password.
    """
    try:
        user_id = int(input("Enter the user ID to update: "))
    except ValueError:
        print("Invalid ID. Please enter a number.")
        return

    new_username = input("Enter new username: ")
    new_password = getpass.getpass("Enter new password: ")
    new_city = input("Enter new city name (optional): ")
    new_password_hash = hash_password(new_password)
    
    with conn:
        cursor = conn.execute(
            "UPDATE users SET username = ?, password_hash = ?, city_name = ? WHERE id = ?",
            (new_username, new_password_hash, new_city, user_id)
        )
    if cursor.rowcount > 0:
        print("User updated successfully.")
    else:
        print("User not found.")

def delete_user(conn):
    """
    Deletes a user from the database.
    """
    try:
        user_id = int(input("Enter the user ID to delete: "))
    except ValueError:
        print("Invalid ID. Please enter a number.")
        return

    with conn:
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    if cursor.rowcount > 0:
        print("User deleted successfully.")
    else:
        print("User not found.")

def main():
    """
    Main function that connects to the database and handles the interactive menu.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        create_table(conn)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    while True:
        print("\nUser Database Management")
        print("1. List users")
        print("2. Add user")
        print("3. Update user")
        print("4. Delete user")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            list_users(conn)
        elif choice == '2':
            add_user(conn)
        elif choice == '3':
            update_user(conn)
        elif choice == '4':
            delete_user(conn)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

    conn.close()

if __name__ == "__main__":
    main() 
