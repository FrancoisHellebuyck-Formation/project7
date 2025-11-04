# Architecture du Projet - Puls-Events

## 📋 Vue d'ensemble

Puls-Events est un système de recherche sémantique et de chatbot conversationnel pour les événements culturels en Occitanie. Le projet combine récupération de données (API OpenAgenda), traitement NLP (chunking, embeddings), recherche vectorielle (FAISS) et génération de réponses (Mistral AI).

## 🏗️ Architecture globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE PULS-EVENTS                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  OpenAgenda API  │
│  (External)      │
└────────┬─────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ get_corpus_      │         │ get_corpus_      │                 │
│  │ agendas.py       │────────▶│ events.py        │                 │
│  │ (Step 1)         │         │ (Step 2)         │                 │
│  └────────┬─────────┘         └────────┬─────────┘                 │
└───────────┼──────────────────────────┼─────────────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                      MongoDB (Docker)                                │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ Collection:      │         │ Collection:      │                 │
│  │ agendas          │         │ events           │                 │
│  └──────────────────┘         └────────┬─────────┘                 │
└──────────────────────────────────────┼─────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐                                               │
│  │ chunks_          │  LangChain Documents                          │
│  │ document.py      │  (1500 chars, 200 overlap)                   │
│  │ (Step 3)         │                                               │
│  └────────┬─────────┘                                               │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ embeddings.py    │  Multilingual-E5-Large                        │
│  │ (Step 4)         │  (1024 dimensions)                            │
│  └────────┬─────────┘                                               │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ vectors.py       │  FAISS Index Creation                         │
│  │ pipeline.py      │  Similarity Search                            │
│  │ (Step 5)         │                                               │
│  └────────┬─────────┘                                               │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE                                    │
├─────────────────────────────────────────────────────────────────────┤
│              FAISS Index (data/faiss_index/)                         │
│              - 28,962+ vecteurs (événements)                         │
│              - Dimension: 1024                                       │
│              - Distance: L2                                          │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐           │
│  │              FastAPI Server (api/main.py)            │           │
│  │              Port: 8000                              │           │
│  ├──────────────────────────────────────────────────────┤           │
│  │ Endpoints:                                           │           │
│  │  • POST /search   - Recherche sémantique            │           │
│  │  • POST /ask      - RAG + Mistral AI                │           │
│  │  • GET  /health   - Health check                    │           │
│  │  • GET  /stats    - Statistics                      │           │
│  └───────────┬──────────────────────────────────────────┘           │
└──────────────┼──────────────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌─────────┐ ┌──────┐ ┌────────────┐
│ CLI     │ │ API  │ │ Streamlit  │
│ Script  │ │Client│ │ Web UI     │
│mistral  │ │      │ │chatbot.py  │
│.py      │ │      │ │Port: 8501  │
└─────────┘ └──────┘ └────────────┘
```

## 🔄 Flux de données détaillés

### 1. Pipeline de collecte des données

```
OpenAgenda API
      │
      ├─► GET /agendas?region=Occitanie
      │   └─► get_corpus_agendas.py
      │       └─► MongoDB.agendas (upsert)
      │
      └─► GET /agendas/{uid}/events
          └─► get_corpus_events.py
              └─► MongoDB.events (upsert avec agendaUid)
```

**Caractéristiques :**
- Pagination avec curseur (`after[]`)
- Batch operations (`bulk_write`)
- Idempotent (upsert)
- Gestion d'erreurs robuste

### 2. Pipeline de traitement NLP

```
MongoDB.events
      │
      ├─► load_documents_from_mongodb()
      │   └─► LangChain Documents
      │       ├─► page_content: formatted text
      │       └─► metadata: {title, city, dates, coords, ...}
      │
      ├─► RecursiveCharacterTextSplitter
      │   ├─► chunk_size: 1500
      │   └─► chunk_overlap: 200
      │
      ├─► E5Embeddings.embed_documents()
      │   ├─► Prefix: "passage: "
      │   ├─► Model: multilingual-e5-large
      │   ├─► Average pooling + L2 normalization
      │   └─► Output: 1024D vectors
      │
      └─► FAISS.from_documents()
          └─► IndexFlatL2 (exact search)
```

**Performances :**
- CPU: ~10-30 chunks/sec
- MPS (Apple Silicon): ~50-100 chunks/sec
- CUDA (NVIDIA): ~100-300 chunks/sec

### 3. Flux de recherche sémantique

```
User Query: "festival de jazz à Toulouse"
      │
      ▼
┌──────────────────────────────────────┐
│ POST /search                         │
│ Body: {"query": "...", "k": 5}      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ E5Embeddings.embed_query()          │
│ Prefix: "query: "                   │
│ Output: 1024D vector                │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ FAISS.similarity_search_with_score() │
│ Distance: L2                         │
│ Top-K: 5                            │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Results: [(doc, score), ...]        │
│ - title, content, location          │
│ - metadata (city, dates, coords)    │
│ - L2 distance score                 │
└──────────────────────────────────────┘
```

### 4. Flux RAG (Retrieval Augmented Generation)

```
User Question: "Quel est le meilleur festival de jazz en été ?"
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ POST /ask                                               │
│ Body: {"question": "...", "k": 5}                      │
└────────────┬────────────────────────────────────────────┘
             │
             ├─► 1. Recherche RAG (Vector Store)
             │   └─► Top-K documents pertinents
             │
             ├─► 2. Formatage du contexte
             │   ├─► Titre, ville, dates
             │   ├─► Contenu (max 500 chars/doc)
             │   └─► Score de pertinence
             │
             ├─► 3. Construction du prompt enrichi
             │   ├─► System prompt (ps.md - Puls-Events)
             │   ├─► Contexte RAG formaté
             │   └─► Question utilisateur
             │
             ├─► 4. Appel Mistral AI
             │   ├─► Model: mistral-small-latest
             │   ├─► Temperature: default
             │   └─► Max tokens: auto
             │
             └─► 5. Réponse
                 ├─► answer: texte généré
                 ├─► context_used: documents sources
                 └─► tokens_used: {prompt, completion, total}
```

### 5. Flux Streamlit UI

```
User (Browser :8501)
      │
      ├─► Input: Question dans le chat
      │
      ▼
┌──────────────────────────────────────┐
│ Streamlit App (ui/chatbot.py)       │
├──────────────────────────────────────┤
│ 1. init_session_state()             │
│    └─► messages: []                 │
│    └─► conversation_started: False  │
│                                      │
│ 2. add_message(user, question)      │
│    └─► Ajout à st.session_state     │
│                                      │
│ 3. call_ask_api(question, k=5)      │
│    ├─► POST localhost:8000/ask      │
│    ├─► Timeout: 30s                 │
│    └─► Response: {answer, context}  │
│                                      │
│ 4. display_chat_message()           │
│    ├─► Avatar: 🎭                   │
│    ├─► Réponse formatée             │
│    └─► Détails (tokens, sources)   │
│                                      │
│ 5. add_message(assistant, answer)   │
│    └─► Sauvegarde en historique     │
└──────────────────────────────────────┘
```

## 📦 Structure des modules

### Package `corpus/`
**Responsabilité :** Collecte de données depuis OpenAgenda

```
corpus/
├── get_corpus_agendas.py   # Récupération des agendas
└── get_corpus_events.py    # Récupération des événements
```

### Package `chunks/`
**Responsabilité :** Traitement et découpage des documents

```
chunks/
└── chunks_document.py      # Chunking avec LangChain
    ├── format_event_content()
    ├── extract_metadata()
    └── process_events_to_chunks()
```

### Package `embeddings/`
**Responsabilité :** Génération des embeddings vectoriels

```
embeddings/
└── embeddings.py           # Modèle E5
    ├── E5Embeddings class
    ├── embed_documents()
    └── embed_query()
```

### Package `vectors/`
**Responsabilité :** Gestion du vector store FAISS

```
vectors/
├── vectors.py              # CRUD operations sur FAISS
│   ├── create_vector_store()
│   ├── load_vector_store()
│   ├── search_similar_documents()
│   └── get_vector_store_stats()
└── server.py              # Serveur REPL interactif
```

### Package `api/`
**Responsabilité :** API REST FastAPI

```
api/
├── models.py              # Modèles Pydantic
│   ├── SearchQuery, SearchResult, SearchResponse
│   └── AskQuery, AskResponse
├── main.py                # Application FastAPI
│   ├── POST /search
│   ├── POST /ask
│   ├── GET /health
│   └── GET /stats
└── __init__.py
```

### Package `chat/`
**Responsabilité :** Chatbot CLI et prompts

```
chat/
├── mistral.py             # CLI chatbot
│   ├── search_rag()
│   ├── format_rag_context()
│   └── main()
└── ps.md                  # Prompt système Puls-Events
```

### Package `ui/`
**Responsabilité :** Interface web Streamlit

```
ui/
├── chatbot.py             # Application Streamlit
│   ├── init_session_state()
│   ├── call_ask_api()
│   ├── display_chat_message()
│   └── main()
└── README.md
```

### Module `pipeline.py`
**Responsabilité :** Orchestration complète

```
pipeline.py
└── create_vector_store_pipeline()
    ├── MongoDB → chunks
    ├── chunks → embeddings
    └── embeddings → FAISS
```

## 🔐 Sécurité et bonnes pratiques

### Variables d'environnement

```
.env (non versionné)
├── OA_API_KEY              # OpenAgenda API key
├── MISTRAL_API_KEY         # Mistral AI API key
├── MONGODB_URI             # MongoDB connection string
├── EMBEDDINGS_DEVICE       # cpu, cuda, mps
└── FAISS_INDEX_PATH        # Chemin de l'index
```

### Gestion des erreurs

- **API externe :** Retry avec backoff exponentiel
- **Connexion MongoDB :** Fermeture dans finally block
- **Embeddings :** Détection automatique du device
- **API REST :** HTTPException avec codes appropriés
- **Streamlit :** Messages utilisateur explicites

### Logging

Tous les modules utilisent le module `logging` Python :
```python
logger = logging.getLogger(__name__)
logger.info("Message informatif")
logger.error("Erreur", exc_info=True)
```

## 🚀 Déploiement

### Architecture de déploiement

```
┌──────────────────────────────────────────────┐
│              Load Balancer                   │
│              (nginx/traefik)                 │
└────────────────┬─────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ FastAPI │ │ FastAPI │ │ FastAPI │
│ Worker  │ │ Worker  │ │ Worker  │
│ :8000   │ │ :8001   │ │ :8002   │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│ MongoDB │ │  FAISS  │ │ Mistral  │
│ Cluster │ │  Index  │ │   API    │
│         │ │ (Shared)│ │(External)│
└─────────┘ └─────────┘ └──────────┘
```

### Docker Compose

```yaml
services:
  mongodb:
    image: mongo:7
    ports: ["27017:27017"]
    volumes: ["./data/mongo:/data/db"]

  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [mongodb]
    volumes: ["./data/faiss_index:/app/data/faiss_index"]
    environment:
      - EMBEDDINGS_DEVICE=cpu
```

## 📊 Métriques et monitoring

### Métriques clés

- **Temps de réponse `/search`** : ~100-500ms
- **Temps de réponse `/ask`** : ~2-5s (dont Mistral AI)
- **Taille index FAISS** : ~250MB (28k vecteurs)
- **Mémoire API** : ~2-4GB (modèle embeddings chargé)
- **Tokens moyens `/ask`** : ~1500-2500 tokens

### Logs structurés

```python
logger.info(f"Recherche: '{query}' (k={k})")
logger.info(f"✓ {len(results)} résultats trouvés")
logger.error(f"❌ Erreur: {e}", exc_info=True)
```

## 🔄 Cycle de mise à jour

```
1. Collecte quotidienne (cron)
   └─► make run-agendas && make run-events

2. Re-processing hebdomadaire
   └─► make run-chunks && make run-embeddings

3. Rechargement API (sans downtime)
   └─► docker-compose restart api
```

## 🎯 Points d'extension futurs

1. **Cache Redis** pour les requêtes fréquentes
2. **Elasticsearch** pour recherche full-text combinée
3. **Qdrant/Weaviate** pour vector store distribué
4. **Celery** pour tâches asynchrones
5. **Monitoring** avec Prometheus + Grafana
6. **A/B Testing** des modèles d'embeddings
7. **Fine-tuning** du modèle E5 sur événements culturels
8. **Multi-tenancy** pour autres régions

## 📚 Références techniques

- **FAISS** : https://github.com/facebookresearch/faiss
- **LangChain** : https://python.langchain.com/
- **E5 Embeddings** : https://huggingface.co/intfloat/multilingual-e5-large
- **Mistral AI** : https://docs.mistral.ai/
- **FastAPI** : https://fastapi.tiangolo.com/
- **Streamlit** : https://docs.streamlit.io/
