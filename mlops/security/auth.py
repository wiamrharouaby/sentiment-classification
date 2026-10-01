from typing import Dict, Any, Optional, List
import jwt
from datetime import datetime, timedelta
import logging
from functools import wraps
import hashlib
import os
import base64

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Décorateur pour vérifier les permissions
def require_permissions(required_permissions: List[str]):
    def decorator(func):
        @wraps(func)
        def wrapper(user: Dict[str, Any], *args, **kwargs):
            user_permissions = user.get("permissions", [])
            
            # Vérifier si l'utilisateur a toutes les permissions requises
            if not all(perm in user_permissions for perm in required_permissions):
                raise PermissionError(f"L'utilisateur n'a pas les permissions requises: {required_permissions}")
            
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

class AuthManager:
    def __init__(self, 
                 secret_key: str,
                 algorithm: str = "HS256",
                 token_expiration_minutes: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiration_minutes = token_expiration_minutes
        self.users_db: Dict[str, Dict[str, Any]] = {}
    
    def register_user(self, 
                      username: str, 
                      password: str, 
                      email: Optional[str] = None,
                      full_name: Optional[str] = None,
                      permissions: Optional[List[str]] = None) -> bool:
        """
        Enregistre un nouvel utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            email: Email (optionnel)
            full_name: Nom complet (optionnel)
            permissions: Liste des permissions (optionnel)
            
        Returns:
            True si l'enregistrement a réussi, False sinon
        """
        if username in self.users_db:
            logging.warning(f"L'utilisateur {username} existe déjà")
            return False
        
        # Générer un sel aléatoire
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode('utf-8')
        
        # Hacher le mot de passe avec le sel
        hashed_password = self._hash_password(password, salt)
        
        # Enregistrer l'utilisateur
        self.users_db[username] = {
            "username": username,
            "hashed_password": hashed_password,
            "salt": salt_b64,
            "email": email,
            "full_name": full_name,
            "permissions": permissions or [],
            "disabled": False
        }
        
        logging.info(f"Utilisateur {username} enregistré avec succès")
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authentifie un utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            
        Returns:
            Informations sur l'utilisateur si l'authentification réussit, None sinon
        """
        if username not in self.users_db:
            logging.warning(f"L'utilisateur {username} n'existe pas")
            return None
        
        user = self.users_db[username]
        
        if user.get("disabled", False):
            logging.warning(f"L'utilisateur {username} est désactivé")
            return None
        
        # Récupérer le sel
        salt_b64 = user.get("salt")
        salt = base64.b64decode(salt_b64)
        
        # Hacher le mot de passe avec le sel récupéré
        hashed_password = self._hash_password(password, salt)
        
        # Vérifier si le mot de passe correspond
        if hashed_password != user["hashed_password"]:
            logging.warning(f"Mot de passe incorrect pour l'utilisateur {username}")
            return None
        
        # Retourner les informations de l'utilisateur (sans le mot de passe)
        user_info = user.copy()
        user_info.pop("hashed_password")
        user_info.pop("salt")
        
        return user_info
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """
        Crée un token JWT pour l'utilisateur
        
        Args:
            user_data: Données de l'utilisateur à inclure dans le token
            
        Returns:
            Token JWT
        """
        # Préparer les données à encoder
        to_encode = user_data.copy()
        
        # Ajouter la date d'expiration
        expire = datetime.utcnow() + timedelta(minutes=self.token_expiration_minutes)
        to_encode.update({"exp": expire})
        
        # Encoder le token
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Vérifie un token JWT
        
        Args:
            token: Token JWT à vérifier
            
        Returns:
            Données de l'utilisateur si le token est valide, None sinon
        """
        try:
            # Décoder le token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Vérifier si le token a expiré
            if "exp" in payload and datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                logging.warning("Token expiré")
                return None
            
            return payload
        except jwt.PyJWTError as e:
            logging.warning(f"Erreur lors de la vérification du token: {e}")
            return None
    
    def _hash_password(self, password: str, salt: bytes) -> str:
        """
        Hache un mot de passe avec un sel
        
        Args:
            password: Mot de passe à hacher
            salt: Sel à utiliser
            
        Returns:
            Mot de passe haché
        """
        # Utiliser PBKDF2 avec SHA-256
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # Nombre d'itérations
        )
        
        # Encoder en base64
        return base64.b64encode(key).decode('utf-8')
