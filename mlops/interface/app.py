import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import time

# Configuration de l'application
st.set_page_config(
    page_title="Classification de Texte MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variables globales
API_URL = "http://localhost:8000"
TOKEN = None

# Fonction pour enregistrer un nouvel utilisateur
def register_user(username: str, password: str, email: str = None, full_name: str = None) -> bool:
    """
    Enregistre un nouvel utilisateur
    
    Args:
        username: Nom d'utilisateur
        password: Mot de passe
        email: Email (optionnel)
        full_name: Nom complet (optionnel)
        
    Returns:
        True si l'enregistrement a réussi, False sinon
    """
    try:
        response = requests.post(
            f"{API_URL}/register",
            data={
                "username": username, 
                "password": password,
                "email": email,
                "full_name": full_name
            }
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erreur d'enregistrement: {response.json().get('detail', '')}")
            return False
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {e}")
        return False

# Fonction pour obtenir un token d'authentification
def get_token(username: str, password: str) -> Optional[str]:
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            return data["access_token"]
        else:
            st.error(f"Erreur d'authentification: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {e}")
        return None

# Interface utilisateur
def login_page():
    """Page de connexion"""
    st.title("Classification de Texte MLOps")
    
    # Onglets pour la connexion et l'inscription
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])
    
    # Onglet Connexion
    with tab1:
        st.subheader("Connexion")
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")
            
            if submit:
                if username and password:
                    with st.spinner("Connexion en cours..."):
                        token = get_token(username, password)
                        if token:
                            st.session_state["token"] = token
                            st.session_state["username"] = username
                            st.success("Connexion réussie!")
                            time.sleep(1)
                            st.experimental_rerun()
                else:
                    st.warning("Veuillez remplir tous les champs")
    
    # Onglet Inscription
    with tab2:
        st.subheader("Inscription")
        with st.form("register_form"):
            new_username = st.text_input("Nom d'utilisateur")
            new_password = st.text_input("Mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            email = st.text_input("Email (optionnel)")
            full_name = st.text_input("Nom complet (optionnel)")
            register_submit = st.form_submit_button("S'inscrire")
            
            if register_submit:
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("Les mots de passe ne correspondent pas")
                    else:
                        with st.spinner("Inscription en cours..."):
                            if register_user(new_username, new_password, email, full_name):
                                st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
                else:
                    st.warning("Veuillez remplir tous les champs obligatoires")

# Classification Page (Pas de modification ici)
def classification_page():
    # Reste du code pour cette page
    pass

# Application principale
def main():
    """Point d'entrée principal"""
    if "token" not in st.session_state:
        login_page()
    else:
        classification_page()

if __name__ == "__main__":
    main()
