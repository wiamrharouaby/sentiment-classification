"""
Client API pour la classification de texte
Ce module fournit une interface simple pour interagir avec l'API de classification de texte.
"""

import requests
import json
from typing import Dict, Any, Optional, Tuple

class TextClassificationClient:
    """Client pour l'API de classification de texte"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialise le client API
        
        Args:
            base_url: URL de base de l'API
        """
        self.base_url = base_url
        self.token = None
        self.username = None
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Vérifie si l'API est accessible
        
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                return True, "API accessible"
            else:
                return False, f"API inaccessible. Statut: {response.status_code}"
        except Exception as e:
            return False, f"Erreur de connexion: {str(e)}"
    
    def register(self, username: str, password: str, email: Optional[str] = None, 
                 full_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Inscrit un nouvel utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            email: Email (optionnel)
            full_name: Nom complet (optionnel)
            
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            data = {
                "username": username,
                "password": password
            }
            
            if email:
                data["email"] = email
            if full_name:
                data["full_name"] = full_name
            
            response = requests.post(
                f"{self.base_url}/register",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                return True, "Inscription réussie"
            else:
                try:
                    error_data = response.json()
                    return False, error_data.get("detail", "Échec de l'inscription")
                except:
                    return False, f"Échec de l'inscription. Statut: {response.status_code}"
        except Exception as e:
            return False, f"Erreur lors de l'inscription: {str(e)}"
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Connecte un utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            response = requests.post(
                f"{self.base_url}/login-simple",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=f"username={username}&password={password}"
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.token = data.get("access_token")
                    self.username = username
                    return True, "Connexion réussie"
                else:
                    return False, data.get("message", "Échec de la connexion")
            else:
                return False, f"Échec de la connexion. Statut: {response.status_code}"
        except Exception as e:
            return False, f"Erreur lors de la connexion: {str(e)}"
    
    def classify_text(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Classifie un texte
        
        Args:
            text: Texte à classifier
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (succès, résultat)
        """
        if not self.token:
            return False, {"error": "Non authentifié. Veuillez vous connecter d'abord."}
        
        try:
            response = requests.post(
                f"{self.base_url}/classify",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}"
                },
                data=json.dumps({"text": text})
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, {
                    "sentiment": data.get("sentiment"),
                    "confidence": data.get("confidence"),
                    "is_positive": data.get("sentiment") == "positive"
                }
            else:
                try:
                    error_data = response.json()
                    return False, {"error": error_data.get("detail", "Échec de la classification")}
                except:
                    return False, {"error": f"Échec de la classification. Statut: {response.status_code}"}
        except Exception as e:
            return False, {"error": f"Erreur lors de la classification: {str(e)}"}
    
    def logout(self) -> None:
        """Déconnecte l'utilisateur"""
        self.token = None
        self.username = None
