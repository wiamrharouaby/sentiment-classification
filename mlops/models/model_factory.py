from typing import Dict, Any, Type, Optional, List
from abc import ABC, abstractmethod
import logging
import pickle
import os
from pathlib import Path
import time
from functools import wraps
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Décorateur pour mesurer le temps d'exécution
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Fonction {func.__name__} exécutée en {end_time - start_time:.4f} secondes")
        return result
    return wrapper

# Interface pour les modèles
class TextClassificationModel(ABC):
    @abstractmethod
    def train(self, texts: List[str], labels: List[int]) -> None:
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """Prédit les étiquettes pour les textes donnés"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde le modèle"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Charge le modèle"""
        pass

# Implémentation du modèle SVM
class SVMTextClassifier(TextClassificationModel):
    def __init__(self, 
                 C: float = 1.0, 
                 kernel: str = 'linear',
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 1)):
        self.C = C
        self.kernel = kernel
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model = None
        
    @timing_decorator
    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Entraîne le modèle SVM
        
        Args:
            texts: Liste de textes
            labels: Liste d'étiquettes (0 pour négatif, 1 pour positif)
        """
        # Créer le pipeline avec TF-IDF et SVM
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)),
            ('svm', SVC(C=self.C, kernel=self.kernel, probability=True))
        ])
        
        # Entraîner le modèle
        self.model.fit(texts, labels)
        logging.info("Modèle SVM entraîné avec succès")
    
    @timing_decorator
    def predict(self, texts: List[str]) -> List[int]:
        """
        Prédit les étiquettes pour les textes donnés
        
        Args:
            texts: Liste de textes
            
        Returns:
            Liste d'étiquettes prédites
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        return self.model.predict(texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Prédit les probabilités pour les textes donnés
        
        Args:
            texts: Liste de textes
            
        Returns:
            Tableau de probabilités
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        return self.model.predict_proba(texts)
    
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Modèle sauvegardé à {path}")
    
    def load(self, path: str) -> None:
        """
        Charge le modèle
        
        Args:
            path: Chemin vers le modèle
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logging.info(f"Modèle chargé depuis {path}")

# Factory Pattern pour créer des modèles
class ModelFactory:
    _models: Dict[str, Type[TextClassificationModel]] = {
        "svm": SVMTextClassifier
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[TextClassificationModel]) -> None:
        """
        Enregistre un nouveau type de modèle
        
        Args:
            name: Nom du modèle
            model_class: Classe du modèle
        """
        cls._models[name] = model_class
        logging.info(f"Modèle {name} enregistré")
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> TextClassificationModel:
        """
        Crée une instance de modèle
        
        Args:
            name: Nom du modèle à créer
            **kwargs: Arguments à passer au constructeur du modèle
            
        Returns:
            Instance du modèle
        """
        if name not in cls._models:
            raise ValueError(f"Modèle {name} non enregistré")
        
        return cls._models[name](**kwargs)
