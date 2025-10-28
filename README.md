# OpenClassrooms Project 7 - Cultural Events Search API

API de recherche sémantique pour les événements culturels d'Occitanie, basée sur FastAPI et FAISS.

## Démarrage rapide avec Docker

### Prérequis

- Docker et Docker Compose installés
- L'index FAISS généré dans `data/faiss_index/`

### Lancer l'infrastructure complète

```bash
# Démarrer MongoDB + API
docker-compose up -d

# Vérifier que les services sont démarrés
docker-compose ps

# Voir les logs
docker-compose logs -f api
```

L'API sera disponible sur http://localhost:8000

### Arrêter les services

```bash
docker-compose down
```

### Rebuilder l'image après des modifications

```bash
docker-compose build api
docker-compose up -d api
```

## Utilisation de l'API

### Endpoints disponibles

- `GET /` - Informations sur l'API
- `GET /health` - Health check
- `GET /stats` - Statistiques du vector store
- `GET /search?q=query&k=5` - Recherche sémantique (GET)
- `POST /search` - Recherche sémantique (POST)
- `GET /docs` - Documentation Swagger UI interactive

### Exemples de requêtes

```bash
# Health check
curl http://localhost:8000/health

# Statistiques
curl http://localhost:8000/stats

# Recherche avec GET
curl "http://localhost:8000/search?q=concert+rock&k=5"

# Recherche avec POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "exposition art moderne", "k": 3}'
```

### Documentation interactive

Accédez à http://localhost:8000/docs pour la documentation Swagger UI interactive qui permet de tester directement l'API.

## Développement local sans Docker

### Installation

```bash
# Installer les dépendances
make install

# Ou avec uv directement
uv sync
```

### Commandes disponibles

```bash
make help              # Voir toutes les commandes
make run-all           # Pipeline complet (agendas → events → chunks → embeddings)
make run-api           # Démarrer l'API en mode développement
make docker-up         # Démarrer MongoDB
```

## Architecture

### Pipeline de données

1. **Collecte des agendas** - Récupération depuis OpenAgenda API
2. **Collecte des événements** - Récupération pour chaque agenda
3. **Chunking** - Découpage des documents avec LangChain
4. **Embeddings** - Génération avec multilingual-e5-large (1024D)
5. **Indexation FAISS** - Création de l'index vectoriel
6. **API FastAPI** - Exposition via REST API

### Technologies

- **FastAPI** - Framework web moderne et rapide
- **FAISS** - Vector store pour la recherche sémantique
- **LangChain** - Framework pour le traitement de documents
- **Transformers** - Modèle d'embeddings multilingual-e5-large
- **MongoDB** - Base de données pour les événements bruts
- **Docker** - Containerisation pour le déploiement

## Configuration

Les variables d'environnement sont gérées via le fichier `.env`:

```env
# OpenAgenda API
OA_API_KEY=<your_key>
OA_REGION=Occitanie

# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=OA

# Embeddings
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large
EMBEDDINGS_DEVICE=cpu  # ou cuda, mps
FAISS_INDEX_PATH=data/faiss_index
```

## Production

Pour un déploiement en production:

1. Ajuster le nombre de workers uvicorn dans le Dockerfile
2. Configurer un reverse proxy (nginx, traefik)
3. Activer HTTPS
4. Configurer les limites de rate limiting
5. Monitorer les ressources (CPU, mémoire, GPU si disponible)

## Licence

OpenClassrooms Project 7
