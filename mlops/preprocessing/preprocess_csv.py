import argparse
import pandas as pd
import logging
import os
from pathlib import Path
from functools import wraps
import time

from mlops.preprocessing.text_processor import StandardTextProcessor, BatchTextProcessor

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
    parser = argparse.ArgumentParser(description="Prétraitement des données textuelles CSV")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier CSV d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin vers le fichier CSV de sortie")
    parser.add_argument("--lowercase", action="store_true", help="Convertir en minuscules")
    parser.add_argument("--remove-punctuation", action="store_true", help="Supprimer la ponctuation")
    parser.add_argument("--remove-numbers", action="store_true", help="Supprimer les chiffres")
    
    return parser.parse_args()

@timing_decorator
def preprocess_csv_file(input_path: str, output_path: str, processor: StandardTextProcessor) -> None:
    """
    Prétraite un fichier CSV de données
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        output_path: Chemin vers le fichier de sortie
        processor: Processeur de texte à utiliser
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lire le fichier CSV
# Modifiez cette ligne dans preprocess_csv_file
    df = pd.read_csv(input_path, header=None, names=["label", "text"], sep='\t')    
    # Prétraiter les textes
    df["text"] = df["text"].apply(lambda text: processor.preprocess(text))
    
    # Sauvegarder le fichier prétraité
    df.to_csv(output_path, index=False, header=False)
    
    logging.info(f"Prétraitement terminé. Données sauvegardées à {output_path}")

def main() -> None:
    """Point d'entrée principal"""
    args = setup_argparse()
    
    # Créer le processeur de texte
    processor = StandardTextProcessor(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_numbers=args.remove_numbers
    )
    
    # Prétraiter le fichier
    preprocess_csv_file(args.input, args.output, processor)

if __name__ == "__main__":
    main()