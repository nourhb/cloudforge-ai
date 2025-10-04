import sqlite3
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Create test database
conn = sqlite3.connect('../data/test.db')

# Create sample tables
conn.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.execute('''
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    user_id INTEGER,
    status TEXT DEFAULT 'active',
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

# Insert sample data
users_data = [
    (1, 'John Doe', 'john@example.com'),
    (2, 'Jane Smith', 'jane@example.com'),
    (3, 'Alice Johnson', 'alice@example.com'),
    (4, 'Bob Wilson', 'bob@example.com'),
    (5, 'Charlie Brown', 'charlie@example.com')
]

projects_data = [
    (1, 'CloudForge Migration', 1, 'active'),
    (2, 'AI Analytics Platform', 2, 'completed'),
    (3, 'Database Optimization', 1, 'active'),
    (4, 'Performance Testing', 3, 'pending'),
    (5, 'Security Audit', 2, 'active')
]

conn.executemany('INSERT INTO users (id, name, email) VALUES (?, ?, ?)', users_data)
conn.executemany('INSERT INTO projects (id, name, user_id, status) VALUES (?, ?, ?, ?)', projects_data)

conn.commit()
conn.close()

print("Test database created successfully with sample data")
print("Database path: ../data/test.db")
print("Tables: users (5 records), projects (5 records)")