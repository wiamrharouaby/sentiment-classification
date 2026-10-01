# mlops/database/db.py
import sqlite3
import os
from pathlib import Path
import hashlib
import secrets

# Créer le répertoire de la base de données s'il n'existe pas
os.makedirs("database", exist_ok=True)

def get_db_connection():
    """Établit une connexion à la base de données SQLite"""
    conn = sqlite3.connect("database/users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialise la base de données avec les tables nécessaires"""
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        full_name TEXT,
        hashed_password TEXT NOT NULL,
        salt TEXT NOT NULL,
        disabled BOOLEAN DEFAULT 0
    )
    ''')
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Hache un mot de passe avec un sel"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Utiliser PBKDF2 avec SHA-256
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Nombre d'itérations
    )
    
    return salt, hashlib.sha256(key).hexdigest()

def register_user(username, password, email=None, full_name=None):
    """Enregistre un nouvel utilisateur dans la base de données"""
    conn = get_db_connection()
    
    # Vérifier si l'utilisateur existe déjà
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user:
        conn.close()
        return False, "L'utilisateur existe déjà"
    
    # Hacher le mot de passe
    salt, hashed_password = hash_password(password)
    
    # Insérer l'utilisateur
    conn.execute(
        'INSERT INTO users (username, email, full_name, hashed_password, salt) VALUES (?, ?, ?, ?, ?)',
        (username, email, full_name, hashed_password, salt)
    )
    conn.commit()
    conn.close()
    
    return True, "Utilisateur enregistré avec succès"

def verify_user(username, password):
    """Vérifie les identifiants d'un utilisateur"""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    
    if not user:
        return False, None
    
    salt = user['salt']
    stored_hash = user['hashed_password']
    
    # Vérifier le mot de passe
    _, computed_hash = hash_password(password, salt)
    
    if computed_hash == stored_hash:
        # Convertir Row en dictionnaire
        user_dict = dict(user)
        # Ne pas renvoyer le mot de passe haché et le sel
        user_dict.pop('hashed_password')
        user_dict.pop('salt')
        return True, user_dict
    
    return False, None

# Initialiser la base de données au démarrage
init_db()

# Ajouter des utilisateurs par défaut si la base de données est vide
conn = get_db_connection()
if not conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]:
    register_user("admin", "admin", "admin@example.com", "Administrateur")
    register_user("user", "user", "user@example.com", "Utilisateur Normal")
conn.close()