# 🧠 Pipeline MLOps complet pour la classification de texte (Français / Anglais)

API REST de classification de sentiment (positif / négatif) construite avec **FastAPI** et un modèle **SVM + TF-IDF**, avec authentification JWT, interface web et pipeline MLOps complet (prétraitement, entraînement, évaluation, déploiement Docker).

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

## ✨ Fonctionnalités

- Classification de sentiment de textes en **français et en anglais**
- Score de confiance pour chaque prédiction
- Inscription et connexion des utilisateurs (**JWT**, mots de passe hachés avec **bcrypt**)
- Documentation interactive automatique (Swagger / ReDoc)
- Interface web (inscription, connexion, classification, historique)
- Pipeline MLOps : prétraitement, entraînement, évaluation, métriques, Docker

## 🗂️ Structure du projet

```
mlos/
├── mlops/
│   ├── api/             # API FastAPI (routes, authentification)
│   ├── preprocessing/   # Nettoyage des données
│   ├── models/          # Chargement du modèle (ModelSingleton, factory)
│   ├── evaluation/      # Évaluation et métriques
│   ├── docker/          # Dockerfile et docker-compose
│   ├── monitoring/      # Suivi
│   └── main.py          # Point d'entrée (train / service)
├── models/              # Modèles entraînés (.pkl)
├── metrics/             # Résultats d'évaluation (JSON)
├── train_multilingual.py  # Entraînement du modèle FR + EN
├── interface_moderne.html # Interface web principale
├── inscription.html
├── test_auth.html
└── requirements-simple.txt
```

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/wiamrharouaby/<nom-du-depot>.git
cd <nom-du-depot>
```

### 2. Créer et activer l'environnement virtuel

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
# Installation complète
pip install -r mlops/requirements.txt

# ou installation simplifiée
pip install -r requirements-simple.txt
```

## 🧪 Pipeline de données et d'entraînement

### Option A : modèle multilingue (français + anglais) — recommandé

Le script télécharge automatiquement trois jeux de données publics (Allociné pour le français, IMDB et SST-2 pour l'anglais), entraîne un SVM linéaire calibré et enregistre le modèle :

```bash
pip install datasets
python train_multilingual.py
```

Sortie : `models/svm_model_multi.pkl` (accuracy d'environ 0,89 sur le jeu de test).

### Option B : pipeline classique sur votre propre CSV

```bash
# 1. Prétraitement des données
python -m mlops.preprocessing.preprocess_csv \
  --input mlops/data/raw/reviews.csv \
  --output mlops/data/processed/reviews_processed.csv \
  --lowercase --remove-punctuation

# 2. Entraînement
python -m mlops.main train \
  --data mlops/data/processed/reviews_processed.csv \
  --model-output models/svm_model.pkl \
  --model-type svm

# 3. Évaluation
python -m mlops.evaluation.evaluate \
  --model models/svm_model.pkl \
  --data mlops/data/processed/reviews_processed.csv \
  --output metrics/evaluation_metrics.json
```

> ⚠️ Le modèle doit être un `Pipeline` scikit-learn (vectoriseur + classifieur), car l'API appelle `predict_proba` directement sur du texte brut. Il doit aussi être sauvegardé avec `pickle` (et non `joblib`), format attendu par le chargeur de modèle.

## ▶️ Lancer l'API

```bash
python -m mlops.main service --model models/svm_model_multi.pkl --port 8000
```

| Adresse | Description |
|---|---|
| http://localhost:8000/ | Message de bienvenue |
| http://localhost:8000/docs | Documentation Swagger (tests interactifs) |
| http://localhost:8000/redoc | Documentation ReDoc |
| http://localhost:8000/health | Vérification de l'état du service |

## 🐳 Déploiement avec Docker

```bash
# Démarrer
docker-compose -f mlops/docker/docker-compose.yml up -d

# Reconstruire l'image puis démarrer
docker-compose -f mlops/docker/docker-compose.yml up -d --build
```

## 🌐 Utiliser l'interface web

1. **Démarrez l'API** (voir ci-dessus) et laissez le terminal ouvert.
2. **Ouvrez `interface_moderne.html`** dans votre navigateur (double-clic, ou clic droit → *Open with Live Server* dans VS Code).
3. **Inscription** : onglet « Inscription », remplissez les champs puis cliquez sur « S'inscrire ». Un message de succès s'affiche.
4. **Connexion** : connectez-vous avec le nom d'utilisateur et le mot de passe créés.
5. **Classification** : saisissez un texte (français ou anglais), cliquez sur « Classifier » : le sentiment et la confiance s'affichent.

## 🔌 Endpoints de l'API

| Méthode | Route | Description | Corps de la requête |
|---|---|---|---|
| `POST` | `/register` | Créer un compte | JSON : `username`, `password`, `email`, `full_name` |
| `POST` | `/token` | Obtenir un jeton JWT (valable 30 min) | Formulaire : `username`, `password` |
| `POST` | `/classify` | Classifier un texte | JSON : `{"text": "..."}` |
| `GET` | `/health` | État du service | — |

### Exemple

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Ce produit est vraiment excellent"}'
```

Réponse :

```json
{
  "sentiment": "positive",
  "confidence": 0.97
}
```

## ⚠️ Limites connues

- Le modèle repose sur TF-IDF : il comprend mal les **négations** et les **phrases très courtes** hors vocabulaire d'entraînement.
- Les données d'entraînement sont surtout des critiques de films : les performances peuvent baisser sur d'autres domaines.
- Les utilisateurs sont stockés **en mémoire** : ils sont perdus à chaque redémarrage du service.
- La route `/classify` n'est pas protégée par le jeton JWT.
- `SECRET_KEY` est définie dans le code : **à déplacer dans une variable d'environnement avant toute mise en production**.

## 🛣️ Pistes d'amélioration

- [ ] Stocker les utilisateurs dans SQLite / PostgreSQL
- [ ] Protéger `/classify` avec `Depends(get_current_user)`
- [ ] Charger `SECRET_KEY` depuis une variable d'environnement
- [ ] Ajouter une classe « neutre » (seuil de confiance)
- [ ] Enrichir les données avec des phrases courtes françaises
- [ ] Tester un modèle Transformer multilingue (CamemBERT, mBERT)
- [ ] Ajouter des tests automatisés et une CI GitHub Actions

## 👩‍💻 Auteur

**Wiam Rharouaby** — [GitHub](https://github.com/wiamrharouaby)

 
