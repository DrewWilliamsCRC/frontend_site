import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'CHANGE_ME_TO_SOMETHING_SECURE'  # Replace with your own secret key!

DB_NAME = 'users.db'

def init_db():
    """Create users table if it doesn't exist."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
    print("Database initialized or already exists.")

init_db()

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
        if row:
            stored_hash = row[0]
            if check_password_hash(stored_hash, password):
                session['user'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
        flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("Username and password required.", "danger")
            return redirect(url_for('register'))
        pw_hash = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                            (username, pw_hash))
                conn.commit()
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Choose another.', 'danger')
    return render_template('register.html')

if __name__ == '__main__':
    # Dev server
    app.run(debug=True, host='0.0.0.0', port=5001)
