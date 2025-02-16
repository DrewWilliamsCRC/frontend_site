from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash
from contextlib import contextmanager

user_manager_bp = Blueprint("user_manager", __name__, template_folder="templates")

# Before each request to any admin route, check if the user is logged in.
@user_manager_bp.before_request
def require_login():
    if "user" not in session:
        flash("Please log in to access admin pages.", "warning")
        return redirect(url_for("login"))

DB_NAME = "/app/data/users.db"

@contextmanager
def get_db_connection():
    """Provide a transactional scope around a series of database operations."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@user_manager_bp.route("/users")
def index():
    with get_db_connection() as conn:
        users = conn.execute("SELECT id, username, city_name FROM users").fetchall()
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
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, city_name) VALUES (?, ?, ?)",
                    (username, password_hash, city_name),
                )
                conn.commit()
            flash("User added successfully", "success")
            return redirect(url_for("user_manager.index"))
        except sqlite3.IntegrityError:
            flash("Error: Username already exists.", "danger")
            return redirect(url_for("user_manager.add_user"))
    return render_template("add_user.html")

@user_manager_bp.route("/users/edit/<int:user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
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
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE users SET username = ?, password_hash = ?, city_name = ? WHERE id = ?",
                    (username, password_hash, city_name, user_id),
                )
                conn.commit()
            flash("User updated successfully", "success")
            return redirect(url_for("user_manager.index"))
        except sqlite3.IntegrityError:
            flash("Error: Username conflict or other issue.", "danger")
            return redirect(url_for("user_manager.edit_user", user_id=user_id))
    return render_template("edit_user.html", user=user)

@user_manager_bp.route("/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    flash("User deleted successfully", "success")
    return redirect(url_for("user_manager.index"))
