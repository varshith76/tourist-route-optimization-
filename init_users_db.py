import sqlite3
from werkzeug.security import generate_password_hash

# Create users.db and users table
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')
# Example user: username 'admin', password 'admin123'
hashed_pw = generate_password_hash('admin123')
try:
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', hashed_pw))
except sqlite3.IntegrityError:
    pass  # User already exists
conn.commit()
conn.close()
print('Database and example user created.')
