from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import psycopg2
from werkzeug.security import generate_password_hash
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor

user_manager_bp = Blueprint("user_manager", __name__, template_folder="templates")

# Before each request to any admin route, check if the user is logged in.
@user_manager_bp.before_request
def require_login():
    if "user" not in session:
        flash("Please log in to access admin pages.", "warning")
        return redirect(url_for("login"))

# Read the connection string from environment variables.
if os.environ.get("FLASK_ENV") == "development":
    DATABASE_URL = os.environ.get("DEV_DATABASE_URL", os.environ.get("DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please set it for the environment!")

def get_db_connection():
    # Return a new PostgreSQL connection with RealDictCursor
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

@user_manager_bp.route("/users")
def index():
    conn = get_db_connection()
    try:
        # Use a cursor to execute queries
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, city_name FROM users")
            users = cur.fetchall()
    finally:
        conn.close()
    return render_template("user_manager_index.html", users=users)

@user_manager_bp.route("/users/add", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        # Strip inputs to remove extraneous spaces.
        username = request.form.get("username", "").strip()
        password = request.form.get("password")
        city_name = request.form.get("city_name", "").strip()
        if not username or not password:
            flash("Username and password are required.", "warning")
            return redirect(url_for("user_manager.add_user"))
        password_hash = generate_password_hash(password)
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash, city_name) VALUES (%s, %s, %s)",
                    (username, password_hash, city_name),
                )
                conn.commit()
            flash("User added successfully", "success")
            return redirect(url_for("user_manager.index"))
        except psycopg2.IntegrityError:
            flash("Error: Username already exists.", "danger")
            return redirect(url_for("user_manager.add_user"))
    return render_template("add_user.html")

@user_manager_bp.route("/users/edit/<int:user_id>", methods=["GET", "POST"])
def edit_user(user_id):
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
        city_name = request.form.get("city_name", "").strip()
        password = request.form.get("password", "")
        if password:
            password_hash = generate_password_hash(password)
        else:
            password_hash = user["password_hash"]

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET username = %s, password_hash = %s, city_name = %s WHERE id = %s",
                    (username, password_hash, city_name, user_id),
                )
                conn.commit()
            flash("User updated successfully", "success")
            return redirect(url_for("user_manager.index"))
        except psycopg2.IntegrityError:
            conn.rollback()
            flash("Error: Username conflict or other issue.", "danger")
            return redirect(url_for("user_manager.edit_user", user_id=user_id))
        finally:
            conn.close()
    return render_template("edit_user.html", user=user)

@user_manager_bp.route("/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
    finally:
        conn.close()
    flash("User deleted successfully", "success")
    return redirect(url_for("user_manager.index"))
