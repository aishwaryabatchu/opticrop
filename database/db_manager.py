import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash

DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "opticrops.db")

def get_db_connection():
    """Establishes a connection to the SQLite database and returns the connection object."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database by creating tables if they do not exist and adding default users."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        print(f"Created database directory: {DB_DIR}")

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Create Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Create Predictions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            N REAL NOT NULL,
            P REAL NOT NULL,
            K REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            predicted_crop TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            yield_potential REAL NOT NULL,
            sowing_season TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    ''')

    # 3. Create Suitability Assessments Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suitability_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            crop TEXT NOT NULL,
            N REAL NOT NULL,
            P REAL NOT NULL,
            K REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            compatibility_score REAL NOT NULL,
            suitability_rating TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    ''')

    # 4. Create System Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            user_id INTEGER,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    ''')

    conn.commit()

    # 5. Insert Default Users if table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        print("Inserting default users (Admin, Researcher, Farmer)...")
        default_users = [
            ("admin", generate_password_hash("admin123"), "Admin"),
            ("researcher", generate_password_hash("researcher123"), "Researcher"),
            ("farmer", generate_password_hash("farmer123"), "Farmer")
        ]
        cursor.executemany(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            default_users
        )
        conn.commit()
        print("Default users created successfully.")

    conn.close()

def add_user(username, password, role):
    """Creates a new user account with a hashed password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        log_action("User Registration", user_id, f"Registered username: {username} with role: {role}")
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def get_user_by_id(user_id):
    """Retrieves user row from database by primary key ID."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user

def get_user_by_username(username):
    """Retrieves user row from database by username."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def delete_user(user_id, admin_id):
    """Deletes a user account from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if target is last admin
    cursor.execute("SELECT role, username FROM users WHERE id = ?", (user_id,))
    target = cursor.fetchone()
    if target and target['role'] == 'Admin':
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'Admin'")
        admin_count = cursor.fetchone()[0]
        if admin_count <= 1:
            conn.close()
            return False, "Cannot delete the only administrator in the system."
            
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    log_action("User Deletion", admin_id, f"Deleted user ID: {user_id} (Username: {target['username'] if target else 'Unknown'})")
    return True, "User deleted successfully."

def get_all_users():
    """Retrieves all users."""
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return users

def save_prediction(user_id, n, p, k, temp, hum, ph, rain, crop, confidence, yield_est, season):
    """Saves an ML prediction record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_id, N, P, K, temperature, humidity, ph, rainfall, predicted_crop, confidence_score, yield_potential, sowing_season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, n, p, k, temp, hum, ph, rain, crop, confidence, yield_est, season))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id

def save_suitability_assessment(user_id, crop, n, p, k, temp, hum, ph, rain, score, rating):
    """Saves a suitability assessment record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO suitability_assessments (user_id, crop, N, P, K, temperature, humidity, ph, rainfall, compatibility_score, suitability_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, crop, n, p, k, temp, hum, ph, rain, score, rating))
    conn.commit()
    assess_id = cursor.lastrowid
    conn.close()
    return assess_id

def log_action(action, user_id, details=None):
    """Logs system events and activities."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_logs (action, user_id, details)
        VALUES (?, ?, ?)
    ''', (action, user_id, details))
    conn.commit()
    conn.close()

def get_all_predictions(search_query=None, crop_filter=None):
    """Retrieves predictions, optionally filtered by search text and crop."""
    conn = get_db_connection()
    query = '''
        SELECT p.*, u.username, u.role as user_role 
        FROM predictions p 
        LEFT JOIN users u ON p.user_id = u.id
    '''
    params = []
    conditions = []

    if search_query:
        conditions.append("(u.username LIKE ? OR p.predicted_crop LIKE ?)")
        params.append(f"%{search_query}%")
        params.append(f"%{search_query}%")
    
    if crop_filter:
        conditions.append("p.predicted_crop = ?")
        params.append(crop_filter)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY p.created_at DESC"
    
    predictions = conn.execute(query, params).fetchall()
    conn.close()
    return predictions

def get_user_predictions(user_id, search_query=None):
    """Retrieves predictions for a specific user, optionally filtered by search text."""
    conn = get_db_connection()
    query = 'SELECT * FROM predictions WHERE user_id = ?'
    params = [user_id]
    
    if search_query:
        query += ' AND (predicted_crop LIKE ? OR sowing_season LIKE ?)'
        params.append(f"%{search_query}%")
        params.append(f"%{search_query}%")
        
    query += ' ORDER BY created_at DESC'
    
    predictions = conn.execute(query, params).fetchall()
    conn.close()
    return predictions

def get_system_logs(limit=100):
    """Retrieves recent system logs."""
    conn = get_db_connection()
    logs = conn.execute('''
        SELECT l.*, u.username 
        FROM system_logs l 
        LEFT JOIN users u ON l.user_id = u.id 
        ORDER BY l.created_at DESC 
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return logs

def get_system_analytics_summary():
    """Computes summary statistics for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_preds = cursor.fetchone()[0]

    # Total Users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Most Recommended Crop
    cursor.execute('''
        SELECT predicted_crop, COUNT(*) as cnt 
        FROM predictions 
        GROUP BY predicted_crop 
        ORDER BY cnt DESC 
        LIMIT 1
    ''')
    most_rec_row = cursor.fetchone()
    most_rec_crop = most_rec_row[0] if most_rec_row else "N/A"

    # Average Rainfall
    cursor.execute("SELECT AVG(rainfall) FROM predictions")
    avg_rainfall_val = cursor.fetchone()[0]
    avg_rainfall = round(avg_rainfall_val, 2) if avg_rainfall_val is not None else 0.0

    # Soil Parameter Stats (Average N, P, K, pH)
    cursor.execute("SELECT AVG(N), AVG(P), AVG(K), AVG(ph) FROM predictions")
    stats = cursor.fetchone()
    soil_stats = {
        'avg_N': round(stats[0], 2) if stats[0] is not None else 0.0,
        'avg_P': round(stats[1], 2) if stats[1] is not None else 0.0,
        'avg_K': round(stats[2], 2) if stats[2] is not None else 0.0,
        'avg_ph': round(stats[3], 2) if stats[3] is not None else 0.0
    }

    # Top Crop Predictions Distribution (For Dashboard Charts)
    cursor.execute('''
        SELECT predicted_crop, COUNT(*) as count 
        FROM predictions 
        GROUP BY predicted_crop 
        ORDER BY count DESC 
        LIMIT 5
    ''')
    crop_dist = cursor.fetchall()
    crop_distribution = {row['predicted_crop']: row['count'] for row in crop_dist}

    conn.close()

    return {
        'total_predictions': total_preds,
        'total_users': total_users,
        'most_recommended_crop': most_rec_crop,
        'average_rainfall': avg_rainfall,
        'soil_stats': soil_stats,
        'crop_distribution': crop_distribution
    }

if __name__ == "__main__":
    init_db()
    print("Database opticrops.db initialized successfully.")
