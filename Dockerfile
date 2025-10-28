# Dockerfile pour l'API de consultation du vector store FAISS
#
# Ce Dockerfile crée une image pour l'API FastAPI qui permet d'interroger
# le vector store FAISS contenant les événements culturels d'Occitanie.

FROM python:3.13-slim

# Métadonnées
LABEL maintainer="OpenClassrooms Project 7"
LABEL description="API FastAPI pour la recherche sémantique d'événements culturels"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer uv (gestionnaire de paquets rapide)
RUN pip install uv

# Installer les dépendances Python
# uv sync installe toutes les dépendances depuis pyproject.toml et uv.lock
RUN uv sync --frozen --no-dev

# Ajouter le venv créé par uv au PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copier le code source
COPY src/ ./src/

# Copier les données du vector store (index FAISS)
# Note: En production, monter ce répertoire comme volume pour faciliter les mises à jour
COPY data/faiss_index/ ./data/faiss_index/

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Exposer le port de l'API
EXPOSE 8000

# Variables d'environnement par défaut (peuvent être surchargées)
ENV FAISS_INDEX_PATH=/app/data/faiss_index \
    EMBEDDINGS_MODEL=intfloat/multilingual-e5-large \
    EMBEDDINGS_DEVICE=cpu \
    KMP_DUPLICATE_LIB_OK=TRUE

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
# Utilise uvicorn avec 1 worker par défaut
# En production, augmenter le nombre de workers selon les CPU disponibles
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
