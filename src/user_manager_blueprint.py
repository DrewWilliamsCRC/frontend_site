# User Management Blueprint
# ----------------------
# This module provides a Flask Blueprint for managing user accounts.
# It includes routes and functionality for:
# - Viewing all users
# - Adding new users
# - Editing existing users
# - Deleting users
# All routes require authentication and proper session management.

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import psycopg2
from werkzeug.security import generate_password_hash
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor

# Initialize Blueprint with template directory configuration
user_manager_bp = Blueprint("user_manager", __name__, template_folder="templates")

@user_manager_bp.before_request
def require_login():
    """
    Authentication middleware for all routes in this blueprint.
    
    Checks for active user session before allowing access to any
    user management routes. Redirects to login page if no session exists.
    """
    if "user" not in session:
        flash("Please log in to access admin pages.", "warning")
        return redirect(url_for("login"))

# Database Configuration
# --------------------
# Select appropriate database URL based on environment
if os.environ.get("FLASK_ENV") == "development":
    # When running in Docker, we should use the "db" service name, not localhost
    DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("DEV_DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please set it for the environment!")

def get_db_connection():
    """
    Creates and returns a new database connection.
    
    Returns:
        psycopg2.extensions.connection: Database connection configured with
        RealDictCursor for dictionary-style result access
    """
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

@user_manager_bp.route("/users")
def index():
    """
    Displays a list of all users in the system.
    
    Retrieves basic user information (ID, username, email, city) for all users
    and renders them in the user management index template.
    
    Returns:
        str: Rendered HTML template with user list
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, email, city_name FROM users")
            users = cur.fetchall()
    finally:
        conn.close()
    return render_template("user_manager_index.html", users=users)

@user_manager_bp.route("/users/add", methods=["GET", "POST"])
def add_user():
    """
    Handles user creation through a web form.
    
    GET: Displays the user creation form
    POST: Processes the form submission and creates a new user
    
    Form Fields:
        - username: User's login name (required, unique)
        - email: User's email address (required, unique)
        - password: User's password (required)
        - city_name: User's preferred city (optional)
    
    Returns:
        GET: Rendered HTML template with user creation form
        POST: Redirect to user list on success, back to form on error
    
    Note:
        - Implements proper input sanitization
        - Handles username uniqueness conflicts
        - Uses secure password hashing
    """
    if request.method == "POST":
        # Strip inputs to remove extraneous spaces.
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password")
        city_name = request.form.get("city_name", "").strip()
        if not username or not password or not email:
            flash("Username, email, and password are required.", "warning")
            return redirect(url_for("user_manager.add_user"))
        password_hash = generate_password_hash(password)
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, email, password_hash, city_name) VALUES (%s, %s, %s, %s)",
                    (username, email, password_hash, city_name),
                )
                conn.commit()
            flash("User added successfully", "success")
            return redirect(url_for("user_manager.index"))
        except psycopg2.IntegrityError:
            flash("Error: Username or email already exists.", "danger")
            return redirect(url_for("user_manager.add_user"))
    return render_template("add_user.html")

@user_manager_bp.route("/users/edit/<user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    """
    Handles user information updates through a web form.
    
    GET: Displays the user edit form with current values
    POST: Processes the form submission and updates the user
    
    Args:
        user_id (str): UUID of the user to edit
    
    Form Fields:
        - username: User's login name (required, unique)
        - email: User's email address (required, unique)
        - password: User's new password (optional)
        - city_name: User's preferred city (optional)
    
    Returns:
        GET: Rendered HTML template with user edit form
        POST: Redirect to user list on success, back to form on error
    
    Note:
        - Only updates password if new one is provided
        - Maintains existing password hash if no new password given
        - Handles username uniqueness conflicts
        - Implements proper transaction management
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
    finally:
        conn.close()

    if user is None:
        flash("User not found", "danger")
        return redirect(url_for("user_manager.index"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        city_name = request.form.get("city_name", "").strip()
        password = request.form.get("password", "")
        if not username or not email:
            flash("Username and email are required.", "warning")
            return redirect(url_for("user_manager.edit_user", user_id=user_id))
        if password:
            password_hash = generate_password_hash(password)
        else:
            password_hash = user["password_hash"]

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET username = %s, email = %s, password_hash = %s, city_name = %s WHERE id = %s",
                    (username, email, password_hash, city_name, user_id),
                )
                conn.commit()
            flash("User updated successfully", "success")
            return redirect(url_for("user_manager.index"))
        except psycopg2.IntegrityError:
            flash("Error: Username or email already exists.", "danger")
            return redirect(url_for("user_manager.edit_user", user_id=user_id))
        finally:
            conn.close()
    return render_template("edit_user.html", user=user)

@user_manager_bp.route("/users/delete/<user_id>", methods=["POST"])
def delete_user(user_id):
    """
    Handles user deletion requests.
    
    Args:
        user_id (str): UUID of the user to delete
    
    Returns:
        Redirect to user list with success message
    
    Note:
        - Only accepts POST requests for safety
        - Associated data (e.g., custom services) are deleted via CASCADE
        - Implements proper transaction management
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
    finally:
        conn.close()
    flash("User deleted successfully", "success")
    return redirect(url_for("user_manager.index"))
