# OpenClassrooms Project 7 - Puls-Events

Système de recherche sémantique et chatbot conversationnel pour les événements culturels d'Occitanie, combinant RAG (Retrieval Augmented Generation) et Mistral AI.

> 📖 **Documentation complète** : Consultez [ARCHITECTURE.md](ARCHITECTURE.md) pour une vue détaillée de l'architecture et [rapport/technique.md](rapport/technique.md) pour une analyse approfondie.

## Démarrage rapide avec Docker

### Prérequis

- Docker et Docker Compose installés
- Une clé API OpenAgenda et Mistral AI.
- Un fichier `.env` configuré à la racine du projet. Vous pouvez utiliser le template :
  ```bash
  cp .env.example .env
  ```
  Puis, remplissez les clés API (`OA_API_KEY`, `MISTRAL_API_KEY`).

### Lancer l'infrastructure complète

```bash
# 1. Construire l'index vectoriel (si non existant)
# Cette commande va télécharger les données, les traiter et créer l'index FAISS.
make run-all

# 2. Démarrer les services (API + MongoDB)
docker-compose up -d --build

# Vérifier que les services sont démarrés
docker-compose ps

# Consulter les logs de l'API
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

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Informations sur l'API |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Statistiques du vector store |
| `POST` | `/search` | Recherche sémantique |
| `POST` | `/ask` | Question-réponse avec RAG + Mistral AI |
| `GET` | `/docs` | Documentation Swagger UI interactive |

### Exemples de requêtes

```bash
# Health check
curl http://localhost:8000/health

# Statistiques
curl http://localhost:8000/stats

# Recherche sémantique
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "concert de musique", "k": 5}'

# Question avec RAG + Mistral AI
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels festivals de jazz en été ?", "k": 5}'
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
make run-api           # Démarrer l'API REST
make run-ui            # Démarrer l'interface Streamlit
make run-chat          # Démarrer le chatbot CLI
make docker-up         # Démarrer MongoDB
```

## Architecture

> 📖 Consultez [ARCHITECTURE.md](ARCHITECTURE.md) pour la documentation complète incluant :
> - Schémas détaillés des flux de données
> - Architecture de déploiement
> - Structure des modules
> - Points d'extension futurs

### Pipeline de données (résumé)

```
OpenAgenda API → MongoDB → Chunking → Embeddings → FAISS Index
                                                         ↓
                                                    FastAPI
                                                         ↓
                                          ┌──────────────┼──────────┐
                                          ▼                         ▼
                                     CLI Script                API Client
```

### Technologies principales

| Composant | Technologie | Usage |
|-----------|-------------|-------|
| **API** | FastAPI | REST API endpoints |
| **Vector Store** | FAISS | Recherche sémantique |
| **NLP** | LangChain | Document processing |
| **Embeddings** | multilingual-e5-large | 1024D vectors |
| **LLM** | Mistral AI | RAG responses |
| **Database** | MongoDB | Raw events storage |
| **UI** | Streamlit | Web interface |
| **Deploy** | Docker | Containerization |

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
