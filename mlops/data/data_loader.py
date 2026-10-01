from typing import Iterator, List, Dict, Optional, Generator, Any
import os
import json
import csv
import pandas as pd
from pathlib import Path
import logging
from functools import wraps
import time
from abc import ABC, abstractmethod

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

# Décorateur pour logger les appels de fonction
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Appel de la fonction {func.__name__} avec args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper

# Interface pour les chargeurs de données
class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """Charge les données et retourne un itérateur"""
        pass

# Implémentation concrète pour charger des données textuelles
class TextDataLoader(DataLoader):
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Le chemin {data_path} n'existe pas")
    
    @timing_decorator
    @log_function_call
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """Charge les données textuelles par lots"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    
                    if len(batch) >= self.batch_size:
                        yield {"batch": batch}
                        batch = []
                except json.JSONDecodeError:
                    logging.warning(f"Impossible de décoder la ligne: {line}")
            
            # Retourner le dernier lot s'il n'est pas vide
            if batch:
                yield {"batch": batch}
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Rend la classe itérable"""
        return self.load_data()

# Nouvelle classe pour charger des données CSV
class CSVDataLoader(DataLoader):
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Le chemin {data_path} n'existe pas")
    
    @timing_decorator
    @log_function_call
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """Charge les données CSV par lots"""
        # Utiliser pandas pour lire le CSV
        df = pd.read_csv(self.data_path, header=None, names=["label", "text"])
        
        # Traiter par lots
        total_rows = len(df)
        for i in range(0, total_rows, self.batch_size):
            batch_df = df.iloc[i:min(i+self.batch_size, total_rows)]
            batch = [{"text": row["text"], "label": int(row["label"])} 
                    for _, row in batch_df.iterrows()]
            yield {"batch": batch}
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Rend la classe itérable"""
        return self.load_data()

# Générateur pour charger les données de manière paresseuse
def lazy_text_loader(file_path: str, batch_size: int = 32) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Générateur qui charge les données textuelles de manière paresseuse
    
    Args:
        file_path: Chemin vers le fichier de données
        batch_size: Taille des lots à retourner
        
    Yields:
        Liste de dictionnaires contenant les données
    """
    # Vérifier l'extension du fichier
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError:
                    logging.warning(f"Impossible de décoder la ligne: {line}")
            
            # Retourner le dernier lot s'il n'est pas vide
            if batch:
                yield batch
    elif file_path.endswith('.csv'):
        # Lire le CSV par morceaux pour économiser la mémoire
        for chunk in pd.read_csv(file_path, header=None, names=["label", "text"], chunksize=batch_size):
            batch = [{"text": row["text"], "label": int(row["label"])} 
                    for _, row in chunk.iterrows()]
            yield batch
    else:
        raise ValueError(f"Format de fichier non pris en charge: {file_path}")