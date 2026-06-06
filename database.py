import sqlite3

conn = sqlite3.connect("predictions.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex TEXT,
    bmi REAL,
    smoker TEXT,
    children INTEGER,
    region TEXT,
    prediction REAL
)
""")

conn.commit()
conn.close()

print("Database created successfully!")