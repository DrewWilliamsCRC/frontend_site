import os
from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash

app = Flask(__name__)
# Retrieve the secret key from an environment variable.
app.secret_key = os.environ.get("SECRET_KEY", "default_key_for_dev")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set.")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

@app.route('/')
def index():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id, username, city_name FROM users;')
            users = cur.fetchall()
    return render_template('user_manager_index.html', users=users)

@app.route('/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        city_name = request.form.get('city_name', '')
        if not username or not password:
            flash('Username and password are required.', 'warning')
            return redirect(url_for('add_user'))
        password_hash = generate_password_hash(password)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, city_name) VALUES (%s, %s, %s);",
                        (username, password_hash, city_name)
                    )
                    conn.commit()
            flash("User added successfully", "success")
            return redirect(url_for('index'))
        except psycopg2.IntegrityError:
            flash("Error: Username already exists.", "danger")
            return redirect(url_for('add_user'))
    return render_template('add_user.html')

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s;", (user_id,))
        user = cur.fetchone()
    if user is None:
        conn.close()
        flash("User not found", "danger")
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        city_name = request.form.get('city_name', '')
        password = request.form.get('password', '')
        if password:
            password_hash = generate_password_hash(password)
        else:
            password_hash = user["password_hash"]
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET username = %s, password_hash = %s, city_name = %s WHERE id = %s;",
                    (username, password_hash, city_name, user_id)
                )
                conn.commit()
            flash("User updated successfully", "success")
            return redirect(url_for('index'))
        except psycopg2.IntegrityError:
            conn.rollback()
            flash("Error: Username conflict or other issue.", "danger")
            return redirect(url_for('edit_user', user_id=user_id))
    conn.close()
    return render_template('edit_user.html', user=user)

@app.route('/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
            conn.commit()
    flash("User deleted successfully", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 