from typing import Dict, Any, List, Optional, Callable
import time
import logging
import numpy as np
from datetime import datetime
from functools import wraps
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

# Interface pour les moniteurs
class Monitor(ABC):
    @abstractmethod
    def record(self, data: Dict[str, Any]) -> None:
        """Enregistre les données de monitoring"""
        pass
    
    @abstractmethod
    def check_drift(self) -> bool:
        """Vérifie s'il y a une dérive dans les données"""
        pass

# Implémentation d'un moniteur de performances
class PerformanceMonitor(Monitor):
    def __init__(self, 
                 metrics: List[str] = ["accuracy", "latency"],
                 window_size: int = 100,
                 drift_threshold: float = 0.1):
        self.metrics = metrics
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history: Dict[str, List[float]] = {metric: [] for metric in metrics}
        self.baseline: Dict[str, float] = {}
    
    @timing_decorator
    def record(self, data: Dict[str, Any]) -> None:
        """
        Enregistre les métriques de performance
        
        Args:
            data: Dictionnaire contenant les métriques à enregistrer
        """
        timestamp = datetime.now().isoformat()
        
        for metric in self.metrics:
            if metric in data:
                self.history[metric].append(data[metric])
                # Garder seulement les window_size dernières valeurs
                if len(self.history[metric]) > self.window_size:
                    self.history[metric].pop(0)
        
        # Enregistrer les données pour Prometheus (simulation)
        logging.info(f"[{timestamp}] Métriques enregistrées: {data}")
    
    def set_baseline(self) -> None:
        """Définit la ligne de base pour les métriques"""
        for metric in self.metrics:
            if self.history[metric]:
                self.baseline[metric] = np.mean(self.history[metric])
                logging.info(f"Ligne de base définie pour {metric}: {self.baseline[metric]}")
    
    @timing_decorator
    def check_drift(self) -> bool:
        """
        Vérifie s'il y a une dérive dans les performances
        
        Returns:
            True s'il y a une dérive, False sinon
        """
        if not self.baseline:
            logging.warning("Aucune ligne de base définie, impossible de vérifier la dérive")
            return False
        
        drift_detected = False
        
        for metric in self.metrics:
            if not self.history[metric]:
                continue
            
            # Calculer la moyenne récente
            recent_mean = np.mean(self.history[metric][-min(len(self.history[metric]), 10):])
            
            # Calculer la différence relative
            relative_diff = abs(recent_mean - self.baseline[metric]) / self.baseline[metric]
            
            if relative_diff > self.drift_threshold:
                logging.warning(f"Dérive détectée pour {metric}: {relative_diff:.4f} > {self.drift_threshold}")
                drift_detected = True
        
        return drift_detected

# Implémentation d'un moniteur de dérive de données
class DataDriftMonitor(Monitor):
    def __init__(self, 
                 features: List[str],
                 window_size: int = 1000,
                 drift_threshold: float = 0.2):
        self.features = features
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_data: Dict[str, List[float]] = {feature: [] for feature in features}
        self.current_data: Dict[str, List[float]] = {feature: [] for feature in features}
    
    @timing_decorator
    def record(self, data: Dict[str, Any]) -> None:
        """
        Enregistre les caractéristiques des données
        
        Args:
            data: Dictionnaire contenant les caractéristiques à enregistrer
        """
        for feature in self.features:
            if feature in data:
                self.current_data[feature].append(data[feature])
                # Garder seulement les window_size dernières valeurs
                if len(self.current_data[feature]) > self.window_size:
                    self.current_data[feature].pop(0)
    
    def set_reference(self, reference_data: Dict[str, List[float]]) -> None:
        """
        Définit les données de référence
        
        Args:
            reference_data: Dictionnaire contenant les données de référence
        """
        for feature in self.features:
            if feature in reference_data:
                self.reference_data[feature] = reference_data[feature][:self.window_size]
                logging.info(f"Données de référence définies pour {feature}: {len(self.reference_data[feature])} points")
    
    @timing_decorator
    def check_drift(self) -> bool:
        """
        Vérifie s'il y a une dérive dans les données
        
        Returns:
            True s'il y a une dérive, False sinon
        """
        if not any(self.reference_data.values()):
            logging.warning("Aucune donnée de référence définie, impossible de vérifier la dérive")
            return False
        
        drift_detected = False
        
        for feature in self.features:
            if not self.reference_data[feature] or not self.current_data[feature]:
                continue
            
            # Calculer les statistiques de référence
            ref_mean = np.mean(self.reference_data[feature])
            ref_std = np.std(self.reference_data[feature])
            
            # Calculer les statistiques actuelles
            current_mean = np.mean(self.current_data[feature])
            
            # Calculer la différence normalisée
            if ref_std > 0:
                normalized_diff = abs(current_mean - ref_mean) / ref_std
                
                if normalized_diff > self.drift_threshold:
                    logging.warning(f"Dérive détectée pour {feature}: {normalized_diff:.4f} > {self.drift_threshold}")
                    drift_detected = True
        
        return drift_detected

# Stratégie de monitoring (Strategy Pattern)
class MonitoringStrategy(ABC):
    @abstractmethod
    def monitor(self, data: Dict[str, Any]) -> bool:
        """Surveille les données et retourne True s'il y a une alerte"""
        pass

# Implémentation d'une stratégie de monitoring basée sur les performances
class PerformanceMonitoringStrategy(MonitoringStrategy):
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def monitor(self, data: Dict[str, Any]) -> bool:
        """
        Surveille les performances
        
        Args:
            data: Données à surveiller
            
        Returns:
            True s'il y a une alerte, False sinon
        """
        self.monitor.record(data)
        return self.monitor.check_drift()

# Implémentation d'une stratégie de monitoring basée sur les données
class DataDriftMonitoringStrategy(MonitoringStrategy):
    def __init__(self, monitor: DataDriftMonitor):
        self.monitor = monitor
    
    def monitor(self, data: Dict[str, Any]) -> bool:
        """
        Surveille les dérives de données
        
        Args:
            data: Données à surveiller
            
        Returns:
            True s'il y a une alerte, False sinon
        """
        self.monitor.record(data)
        return self.monitor.check_drift()

# Contexte pour utiliser les stratégies de monitoring
class MonitoringContext:
    def __init__(self, strategy: MonitoringStrategy):
        self.strategy = strategy
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def set_strategy(self, strategy: MonitoringStrategy) -> None:
        """
        Change la stratégie de monitoring
        
        Args:
            strategy: Nouvelle stratégie à utiliser
        """
        self.strategy = strategy
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Ajoute un callback pour les alertes
        
        Args:
            callback: Fonction à appeler en cas d'alerte
        """
        self.alert_callbacks.append(callback)
    
    def monitor(self, data: Dict[str, Any]) -> None:
        """
        Surveille les données et déclenche des alertes si nécessaire
        
        Args:
            data: Données à surveiller
        """
        if self.strategy.monitor(data):
            for callback in self.alert_callbacks:
                callback(data)
