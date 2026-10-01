import argparse
import json
import logging
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functools import wraps
import time

from mlops.models.model_singleton import ModelSingleton
from mlops.preprocessing.text_processor import StandardTextProcessor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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

def setup_argparse() -> argparse.Namespace:
    """Configure les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers les données d'évaluation")
    parser.add_argument("--output", type=str, required=True, help="Chemin vers le fichier de sortie des métriques")
    
    return parser.parse_args()

def load_dataset(data_path: str) -> tuple:
    """
    Charge le jeu de données
    
    Args:
        data_path: Chemin vers le fichier de données
        
    Returns:
        Tuple contenant les textes et les étiquettes
    """
    texts = []
    labels = []
    
    # Vérifier l'extension du fichier
    if data_path.endswith('.jsonl'):
        # Code existant pour JSONL
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    texts.append(data["text"])
                    labels.append(data["label"])
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Erreur lors du chargement de la ligne: {e}")
    elif data_path.endswith('.csv'):
        # Nouveau code pour CSV
        df = pd.read_csv(data_path, header=None, names=["label", "text"])
        texts = df["text"].tolist()
        labels = df["label"].tolist()
    else:
        raise ValueError(f"Format de fichier non pris en charge: {data_path}")
    
    return texts, labels

@timing_decorator
def evaluate_model(model_path: str, data_path: str) -> Dict[str, float]:
    """
    Évalue le modèle
    
    Args:
        model_path: Chemin vers le modèle
        data_path: Chemin vers les données d'évaluation
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    # Charger le modèle
    model_singleton = ModelSingleton.get_instance()
    model_singleton.initialize_model("svm")
    model_singleton.load_model(model_path)
    model = model_singleton.get_model()
    
    # Charger les données
    texts, true_labels = load_dataset(data_path)
    
    # Faire les prédictions
    predictions = model.predict(texts)
    probas = model.predict_proba(texts)
    
    # Calculer les métriques
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculer l'AUC (Area Under the Curve)
    # Pour simplifier, nous utilisons la probabilité de la classe positive
    positive_probas = probas[:, 1]
    
    # Calculer le log loss
    epsilon = 1e-15
    log_loss = -np.mean([
        true_labels[i] * np.log(max(positive_probas[i], epsilon)) + 
        (1 - true_labels[i]) * np.log(max(1 - positive_probas[i], epsilon))
        for i in range(len(true_labels))
    ])
    
    # Préparer les métriques
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "log_loss": float(log_loss)
    }
    
    return metrics

def main() -> None:
    """Point d'entrée principal"""
    args = setup_argparse()
    
    # Évaluer le modèle
    logging.info(f"Évaluation du modèle {args.model} sur les données {args.data}")
    metrics = evaluate_model(args.model, args.data)
    
    # Afficher les métriques
    logging.info(f"Métriques d'évaluation:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            logging.info(f"  {metric}: {value:.4f}")
    
    # Sauvegarder les métriques
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Métriques sauvegardées à {args.output}")

if __name__ == "__main__":
    main()