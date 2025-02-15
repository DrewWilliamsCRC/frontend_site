from flask import Blueprint, render_template, request, redirect, url_for, flash
import sqlite3
from werkzeug.security import generate_password_hash

user_manager_bp = Blueprint("user_manager", __name__, template_folder="templates")

DB_NAME = "/app/data/users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

@user_manager_bp.route("/users")
def index():
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, city_name FROM users").fetchall()
    conn.close()
    return render_template("user_manager_index.html", users=users)

@user_manager_bp.route("/users/add", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        city_name = request.form.get("city_name", "")
        if not username or not password:
            flash("Username and password are required.", "warning")
            return redirect(url_for("user_manager.add_user"))
        password_hash = generate_password_hash(password)
        try:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO users (username, password_hash, city_name) VALUES (?, ?, ?)",
                (username, password_hash, city_name),
            )
            conn.commit()
            conn.close()
            flash("User added successfully", "success")
            return redirect(url_for("user_manager.index"))
        except sqlite3.IntegrityError:
            flash("Error: Username already exists.", "danger")
            return redirect(url_for("user_manager.add_user"))
    return render_template("add_user.html")

@user_manager_bp.route("/users/edit/<int:user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        conn.close()
        flash("User not found", "danger")
        return redirect(url_for("user_manager.index"))
    
    if request.method == "POST":
        username = request.form.get("username")
        city_name = request.form.get("city_name", "")
        password = request.form.get("password", "")
        if password:
            password_hash = generate_password_hash(password)
        else:
            password_hash = user["password_hash"]
        try:
            conn.execute(
                "UPDATE users SET username = ?, password_hash = ?, city_name = ? WHERE id = ?",
                (username, password_hash, city_name, user_id),
            )
            conn.commit()
            conn.close()
            flash("User updated successfully", "success")
            return redirect(url_for("user_manager.index"))
        except sqlite3.IntegrityError:
            flash("Error: Username conflict or other issue.", "danger")
            conn.close()
            return redirect(url_for("user_manager.edit_user", user_id=user_id))
    conn.close()
    return render_template("edit_user.html", user=user)

@user_manager_bp.route("/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted successfully", "success")
    return redirect(url_for("user_manager.index"))
