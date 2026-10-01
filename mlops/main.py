import logging
import argparse
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import pandas as pd

from mlops.data.data_loader import TextDataLoader, lazy_text_loader, CSVDataLoader
from mlops.preprocessing.text_processor import StandardTextProcessor, BatchTextProcessor
from mlops.models.model_factory import ModelFactory
from mlops.models.model_singleton import ModelSingleton
from mlops.monitoring.monitoring import (
    PerformanceMonitor, 
    DataDriftMonitor, 
    PerformanceMonitoringStrategy,
    DataDriftMonitoringStrategy,
    MonitoringContext
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops.log"),
        logging.StreamHandler()
    ]
)

def setup_argparse() -> argparse.Namespace:
    """Configure les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="MLOps pour la classification de texte")
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande d'entraînement
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument("--data", type=str, required=True, help="Chemin vers les données d'entraînement")
    train_parser.add_argument("--model-output", type=str, required=True, help="Chemin où sauvegarder le modèle")
    train_parser.add_argument("--model-type", type=str, default="svm", help="Type de modèle à entraîner")
    
    # Commande de prédiction
    predict_parser = subparsers.add_parser("predict", help="Faire des prédictions")
    predict_parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle")
    predict_parser.add_argument("--input", type=str, required=True, help="Texte ou chemin vers un fichier de textes")
    predict_parser.add_argument("--output", type=str, help="Chemin où sauvegarder les prédictions")
    
    # Commande de service
    service_parser = subparsers.add_parser("service", help="Démarrer le service API")
    service_parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle")
    service_parser.add_argument("--host", type=str, default="127.0.0.1", help="Hôte du service")    
    service_parser.add_argument("--port", type=int, default=8000, help="Port du service")
    
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

def train_model(args: argparse.Namespace) -> None:
    """
    Entraîne le modèle
    
    Args:
        args: Arguments de ligne de commande
    """
    logging.info(f"Chargement des données depuis {args.data}")
    texts, labels = load_dataset(args.data)
    
    logging.info(f"Prétraitement des textes")
    text_processor = StandardTextProcessor(lowercase=True, remove_punctuation=True)
    batch_processor = BatchTextProcessor(text_processor)
    processed_texts = [text_processor.preprocess(text) for text in texts]
    
    logging.info(f"Initialisation du modèle {args.model_type}")
    model_singleton = ModelSingleton.get_instance()
    model_singleton.initialize_model(args.model_type)
    model = model_singleton.get_model()
    
    logging.info(f"Entraînement du modèle")
    model.train(processed_texts, labels)
    
    logging.info(f"Sauvegarde du modèle à {args.model_output}")
    model.save(args.model_output)
    
    logging.info("Entraînement terminé avec succès")

def predict(args: argparse.Namespace) -> None:
    """
    Fait des prédictions
    
    Args:
        args: Arguments de ligne de commande
    """
    logging.info(f"Chargement du modèle depuis {args.model}")
    model_singleton = ModelSingleton.get_instance()
    model_singleton.initialize_model("svm")
    model_singleton.load_model(args.model)
    model = model_singleton.get_model()
    
    text_processor = StandardTextProcessor(lowercase=True, remove_punctuation=True)
    
    # Vérifier si l'entrée est un fichier ou un texte direct
    if os.path.isfile(args.input):
        logging.info(f"Chargement des textes depuis {args.input}")
        texts = []
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input, header=None, names=["label", "text"])
            texts = df["text"].tolist()
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                for line in f:
                    texts.append(line.strip())
    else:
        logging.info("Utilisation du texte fourni directement")
        texts = [args.input]
    
    # Prétraiter les textes
    processed_texts = [text_processor.preprocess(text) for text in texts]
    
    # Faire les prédictions
    logging.info("Prédiction des sentiments")
    predictions = model.predict(processed_texts)
    probas = model.predict_proba(processed_texts)
    
    # Préparer les résultats
    results = []
    for i, (text, pred, proba) in enumerate(zip(texts, predictions, probas)):
        sentiment = "positif" if pred == 1 else "négatif"
        confidence = proba[1] if pred == 1 else proba[0]
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": float(confidence)
        })
        logging.info(f"Texte {i+1}: {sentiment} (confiance: {confidence:.4f})")
    
    # Sauvegarder les résultats si demandé
    if args.output:
        logging.info(f"Sauvegarde des résultats à {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info("Prédiction terminée avec succès")

def start_service(args: argparse.Namespace) -> None:
    """
    Démarre le service API
    
    Args:
        args: Arguments de ligne de commande
    """
    import uvicorn
    from mlops.api.api import app
    
    logging.info(f"Chargement du modèle depuis {args.model}")
    model_singleton = ModelSingleton.get_instance()
    model_singleton.initialize_model("svm")
    model_singleton.load_model(args.model)
    
    logging.info(f"Démarrage du service sur {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

def main() -> None:
    """Point d'entrée principal"""
    args = setup_argparse()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "service":
        start_service(args)
    else:
        logging.error("Commande non reconnue")

if __name__ == "__main__":
    main()