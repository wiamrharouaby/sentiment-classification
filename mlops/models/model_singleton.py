from typing import Optional, Dict, Any
import logging
from mlops.models.model_factory import TextClassificationModel, ModelFactory

# Singleton Pattern pour le modèle
class ModelSingleton:
    _instance: Optional['ModelSingleton'] = None
    _model: Optional[TextClassificationModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ModelSingleton':
        """
        Retourne l'instance unique du singleton
        
        Returns:
            Instance du singleton
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize_model(self, model_type: str, **kwargs) -> None:
        """
        Initialise le modèle
        
        Args:
            model_type: Type de modèle à créer
            **kwargs: Arguments à passer au constructeur du modèle
        """
        if self._model is not None:
            logging.warning("Le modèle est déjà initialisé, il sera réinitialisé")
        
        self._model = ModelFactory.create_model(model_type, **kwargs)
        logging.info(f"Modèle {model_type} initialisé")
    
    def get_model(self) -> TextClassificationModel:
        """
        Retourne le modèle
        
        Returns:
            Instance du modèle
        """
        if self._model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
        
        return self._model
    
    def load_model(self, path: str) -> None:
        """
        Charge le modèle depuis un fichier
        
        Args:
            path: Chemin vers le fichier du modèle
        """
        if self._model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
        
        self._model.load(path)
        logging.info(f"Modèle chargé depuis {path}")
    
    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle dans un fichier
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        if self._model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
        
        self._model.save(path)
        logging.info(f"Modèle sauvegardé à {path}")
