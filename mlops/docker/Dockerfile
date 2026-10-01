FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY mlops/requirements.txt .
# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY mlops/ /app/mlops/
COPY models/ /app/models/

# Exposer le port pour l'API
EXPOSE 8000

# Définir les variables d'environnement
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/svm_model.pkl
ENV LOG_LEVEL=INFO

# Commande par défaut
CMD ["python", "-m", "mlops.main", "service", "--model", "${MODEL_PATH}", "--host", "0.0.0.0", "--port", "8000"]
