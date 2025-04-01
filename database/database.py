import sqlite3
from datetime import datetime
import pytz

def init_db():
    conn = sqlite3.connect('data.db')
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
            content REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_singapore_time():
    tz = pytz.timezone('Asia/Singapore')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

def insert_data(table, content):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    timestamp = get_singapore_time()
    c.execute(f'''
        INSERT INTO {table} (content, timestamp) VALUES (?, ?)
    ''', (content, timestamp))
    conn.commit()
    conn.close()

def fetch_latest_data(table):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute(f'''
        SELECT content FROM {table} ORDER BY timestamp DESC LIMIT 1
    ''')
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def fetch_all_data(table):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute(f'''
        SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 10
    ''')
    result = c.fetchall()
    conn.close()
    return result

def fetch_all_data_with_timestamps(table):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute(f'''
        SELECT id, content, timestamp FROM {table} ORDER BY timestamp DESC 
    ''')
    result = c.fetchall()
    conn.close()
    return result