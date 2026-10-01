import argparse
import json
import logging
from typing import List, Dict, Any
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
    parser = argparse.ArgumentParser(description="Prétraitement des données textuelles")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin vers le fichier de sortie")
    parser.add_argument("--lowercase", action="store_true", help="Convertir en minuscules")
    parser.add_argument("--remove-punctuation", action="store_true", help="Supprimer la ponctuation")
    parser.add_argument("--remove-numbers", action="store_true", help="Supprimer les chiffres")
    
    return parser.parse_args()

@timing_decorator
def preprocess_file(input_path: str, output_path: str, processor: StandardTextProcessor) -> None:
    """
    Prétraite un fichier de données
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        output_path: Chemin vers le fichier de sortie
        processor: Processeur de texte à utiliser
    """
    batch_processor = BatchTextProcessor(processor)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Générateur pour traiter les données par lots
    def process_data_generator(file_path: str, batch_size: int = 100):
        batch = []
        with open(file_path, 'r', encoding='utf-8') as f:
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
    
    # Traiter les données et écrire dans le fichier de sortie
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, batch in enumerate(process_data_generator(input_path)):
            processed_batch = batch_processor.process_data_batch(batch, text_key="text")
            
            for item in processed_batch:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            if (i + 1) % 10 == 0:
                logging.info(f"Traité {(i + 1) * 100} éléments")
    
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
    preprocess_file(args.input, args.output, processor)

if __name__ == "__main__":
    main()
