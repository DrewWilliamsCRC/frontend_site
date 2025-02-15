from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
from werkzeug.security import generate_password_hash

app = Flask(__name__)
app.secret_key = 'CHANGE_ME_TO_SOMETHING_SECURE_FOR_ADMIN'
# Use the same database file as used in app.py
DB_NAME = '/app/data/users.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # enables accessing columns by name
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, city_name FROM users').fetchall()
    conn.close()
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
            conn = get_db_connection()
            conn.execute("INSERT INTO users (username, password_hash, city_name) VALUES (?, ?, ?)",
                         (username, password_hash, city_name))
            conn.commit()
            conn.close()
            flash("User added successfully", "success")
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            flash("Error: Username already exists.", "danger")
            return redirect(url_for('add_user'))
    return render_template('add_user.html')

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        conn.close()
        flash("User not found", "danger")
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        city_name = request.form.get('city_name', '')
        password = request.form.get('password', '')
        # If a new password is provided, update the hash; otherwise, keep the existing hash.
        if password:
            password_hash = generate_password_hash(password)
        else:
            password_hash = user['password_hash']
        try:
            conn.execute("UPDATE users SET username = ?, password_hash = ?, city_name = ? WHERE id = ?",
                         (username, password_hash, city_name, user_id))
            conn.commit()
            conn.close()
            flash("User updated successfully", "success")
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            flash("Error: Username conflict or other issue.", "danger")
            conn.close()
            return redirect(url_for('edit_user', user_id=user_id))
    
    conn.close()
    return render_template('edit_user.html', user=user)

@app.route('/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted successfully", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 