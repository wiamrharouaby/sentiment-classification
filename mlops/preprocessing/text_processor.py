from typing import List, Dict, Any, Callable, Optional
import re
import string
from functools import wraps
import logging
from abc import ABC, abstractmethod

# Décorateur pour valider les entrées
def validate_input(func):
    @wraps(func)
    def wrapper(self, text: str, *args, **kwargs):
        if not isinstance(text, str):
            raise TypeError("Le texte doit être une chaîne de caractères")
        if not text.strip():
            logging.warning("Texte vide passé au préprocesseur")
            return ""
        return func(self, text, *args, **kwargs)
    return wrapper

# Interface pour les préprocesseurs de texte
class TextProcessor(ABC):
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Prétraite le texte"""
        pass

# Implémentation concrète d'un préprocesseur de texte
class StandardTextProcessor(TextProcessor):
    def __init__(self, 
                 lowercase: bool = True, 
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 custom_filters: Optional[List[Callable[[str], str]]] = None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.custom_filters = custom_filters or []
    
    @validate_input
    def preprocess(self, text: str) -> str:
        """
        Prétraite le texte selon les options configurées
        
        Args:
            text: Texte à prétraiter
            
        Returns:
            Texte prétraité
        """
        # Appliquer les transformations configurées
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Appliquer les filtres personnalisés
        for filter_func in self.custom_filters:
            text = filter_func(text)
            
        return text.strip()

# Classe pour le traitement par lots
class BatchTextProcessor:
    def __init__(self, processor: TextProcessor):
        self.processor = processor
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Prétraite un lot de textes
        
        Args:
            texts: Liste de textes à prétraiter
            
        Returns:
            Liste de textes prétraités
        """
        return [self.processor.preprocess(text) for text in texts]
    
    def process_data_batch(self, data_batch: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Prétraite les textes dans un lot de données
        
        Args:
            data_batch: Liste de dictionnaires contenant les données
            text_key: Clé pour accéder au texte dans les dictionnaires
            
        Returns:
            Liste de dictionnaires avec les textes prétraités
        """
        processed_batch = []
        for item in data_batch:
            if text_key in item:
                processed_item = item.copy()
                processed_item[text_key] = self.processor.preprocess(item[text_key])
                processed_batch.append(processed_item)
            else:
                logging.warning(f"Clé '{text_key}' non trouvée dans l'élément: {item}")
                processed_batch.append(item)
        
        return processed_batch
