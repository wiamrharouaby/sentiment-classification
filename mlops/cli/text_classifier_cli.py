"""
Interface en ligne de commande pour la classification de texte
Ce module fournit une interface en ligne de commande simple pour interagir avec l'API de classification de texte.
"""

import argparse
import sys
import os
from typing import List, Optional
from mlops.api.api_client import TextClassificationClient

class TextClassifierCLI:
    """Interface en ligne de commande pour la classification de texte"""
    
    def __init__(self):
        """Initialise l'interface en ligne de commande"""
        self.client = TextClassificationClient()
        self.token_file = os.path.join(os.path.expanduser("~"), ".text_classifier_token")
        self._load_token()
    
    def _load_token(self) -> None:
        """Charge le token d'authentification depuis le fichier"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        self.client.token = lines[0].strip()
                        self.client.username = lines[1].strip()
            except Exception:
                pass
    
    def _save_token(self) -> None:
        """Sauvegarde le token d'authentification dans un fichier"""
        if self.client.token and self.client.username:
            try:
                with open(self.token_file, "w") as f:
                    f.write(f"{self.client.token}\n{self.client.username}")
            except Exception:
                pass
    
    def _clear_token(self) -> None:
        """Supprime le token d'authentification"""
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
            except Exception:
                pass
    
    def check_health(self) -> None:
        """Vérifie si l'API est accessible"""
        success, message = self.client.check_health()
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
    
    def register(self, username: str, password: str, email: Optional[str] = None, 
                 full_name: Optional[str] = None) -> None:
        """Inscrit un nouvel utilisateur"""
        success, message = self.client.register(username, password, email, full_name)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
    
    def login(self, username: str, password: str) -> None:
        """Connecte un utilisateur"""
        success, message = self.client.login(username, password)
        if success:
            print(f"✅ {message}")
            self._save_token()
        else:
            print(f"❌ {message}")
    
    def classify(self, text: str) -> None:
        """Classifie un texte"""
        success, result = self.client.classify_text(text)
        if success:
            sentiment = "positif" if result.get("is_positive") else "négatif"
            confidence = result.get("confidence", 0) * 100
            print(f"Résultat: Sentiment {sentiment} (confiance: {confidence:.2f}%)")
        else:
            print(f"❌ {result.get('error', 'Erreur inconnue')}")
    
    def logout(self) -> None:
        """Déconnecte l'utilisateur"""
        self.client.logout()
        self._clear_token()
        print("✅ Déconnexion réussie")
    
    def run(self, args: List[str] = None) -> None:
        """Exécute l'interface en ligne de commande"""
        if args is None:
            args = sys.argv[1:]
        
        parser = argparse.ArgumentParser(description="Classification de texte MLOps")
        subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
        
        # Commande health
        health_parser = subparsers.add_parser("health", help="Vérifie si l'API est accessible")
        
        # Commande register
        register_parser = subparsers.add_parser("register", help="Inscrit un nouvel utilisateur")
        register_parser.add_argument("username", help="Nom d'utilisateur")
        register_parser.add_argument("password", help="Mot de passe")
        register_parser.add_argument("--email", help="Email (optionnel)")
        register_parser.add_argument("--full-name", help="Nom complet (optionnel)")
        
        # Commande login
        login_parser = subparsers.add_parser("login", help="Connecte un utilisateur")
        login_parser.add_argument("username", help="Nom d'utilisateur")
        login_parser.add_argument("password", help="Mot de passe")
        
        # Commande classify
        classify_parser = subparsers.add_parser("classify", help="Classifie un texte")
        classify_parser.add_argument("text", help="Texte à classifier")
        
        # Commande logout
        logout_parser = subparsers.add_parser("logout", help="Déconnecte l'utilisateur")
        
        # Analyse les arguments
        parsed_args = parser.parse_args(args)
        
        # Exécute la commande
        if parsed_args.command == "health":
            self.check_health()
        elif parsed_args.command == "register":
            self.register(parsed_args.username, parsed_args.password, 
                         parsed_args.email, parsed_args.full_name)
        elif parsed_args.command == "login":
            self.login(parsed_args.username, parsed_args.password)
        elif parsed_args.command == "classify":
            self.classify(parsed_args.text)
        elif parsed_args.command == "logout":
            self.logout()
        else:
            parser.print_help()

def main():
    """Point d'entrée principal"""
    cli = TextClassifierCLI()
    cli.run()

if __name__ == "__main__":
    main()
