import sqlite3
from datetime import datetime
import pytz
import os

# Always point to the same data.db file in the root project folder
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data.db'))
print("Using database at:", DB_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS accelerometer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS camera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS microphone (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,  -- Changed to TEXT to store emojis + confidence string
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_singapore_time():
    tz = pytz.timezone('Asia/Singapore')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

def insert_data(table, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = get_singapore_time()
    c.execute(f'''
        INSERT INTO {table} (content, timestamp) VALUES (?, ?)
    ''', (content, timestamp))
    conn.commit()
    conn.close()

def fetch_latest_data(table):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f'''
        SELECT content FROM {table} ORDER BY timestamp DESC LIMIT 1
    ''')
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def fetch_all_data(table):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f'''
        SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 10
    ''')
    result = c.fetchall()
    conn.close()
    return result

def fetch_all_data_with_timestamps(table):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f'''
        SELECT id, content, timestamp FROM {table} ORDER BY timestamp DESC LIMIT 10
    ''')
    result = c.fetchall()
    conn.close()
    return result
