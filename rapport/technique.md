# Rapport technique – Assistant intelligent de recommandation d’événements culturels

## 1. Objectifs du projet
### Contexte : 

- Puls-Events est une entreprise technologique innovante spécialisée dans le développement d'une plateforme de recommandations culturelles personnalisées.

- Pour améliorer l'expérience utilisateur et répondre à l'évolution des attentes numériques, Puls-Events souhaite intégrer un assistant intelligent capable de gérer les requêtes des utilisateurs en temps réel.

### Problématique :

- Un système RAG (Retrieval-Augmented Generation) répond aux besoins métier de Puls-Events en résolvant la problématique centrale de l'accès précis et fiable à l'information événementielle par un chatbot.

- Le RAG surmonte les limites des modèles de langage classiques (LLMs) pour fournir une solution à la fois performante et économiquement viable.

### Objectif du POC :

- L'objectif du POC pour Puls-Events est de démontrer de manière concrète et mesurable que la technologie RAG (Retrieval-Augmented Generation), en utilisant LangChain, Mistral et FAISS, est la solution optimale pour alimenter le futur chatbot d'événements culturels.

- Cet objectif se décline en trois axes principaux : 
    - la faisabilité technique, 
    - la valeur métier 
    - et la performance.

1. Démontrer la Faisabilité Technique 🛠️

Il s'agit de prouver que l'intégration des composants clés est fonctionnelle et stable, menant à une solution prête pour l'industrialisation.

Intégration du Pipeline Complet : Prouver la capacité à orchestrer le flux de données de bout en bout : de l'extraction des données d'événements récents via l'API Open Agenda, à leur transformation en embeddings, leur stockage dans l'index FAISS, et leur utilisation par le LLM Mistral via LangChain pour la génération de la réponse.

Portabilité et Déploiement : Valider la capacité à livrer un système standardisé et reproductible grâce à la conteneurisation Docker et à l'exposition via une API REST (FastAPI).

2. Démontrer la Valeur Métier (Pertinence) ✨

L'objectif est de s'assurer que le système répond directement aux besoins de l'utilisateur final et de l'entreprise Puls-Events.

Véracité des Réponses : Démontrer que le RAG élimine les "hallucinations" en basant systématiquement les réponses sur le contexte factuel et à jour des événements (dates, lieux, artistes, genres) extrait d'Open Agenda. C'est la validation de la fiabilité de l'information.

Expérience Utilisateur Améliorée : Prouver que le chatbot peut gérer et répondre avec fluidité à une grande variété de questions en langage naturel, y compris les requêtes sémantiques complexes (e.g., "Je cherche quelque chose de familial le week-end prochain") basées sur le jeu de test annoté.

Efficacité Opérationnelle : Montrer que ce système est plus rentable et plus rapide à actualiser qu'une approche de fine-tuning du LLM, car seule la base vectorielle a besoin d'être mise à jour avec les nouveaux événements.

3. Démontrer la Performance (Qualité et Rapidité) ⚡

Il faut quantifier l'efficacité du système à la fois sur la recherche et la génération.

Performance du Retrieval : Mesurer l'efficacité de FAISS à remonter les fragments de texte pertinents. Le Hit Rate (pourcentage de fois où le bon fragment est dans les top-k résultats) est la métrique clé pour valider que la bonne information est trouvée.

Qualité de la Génération : Mesurer la fidélité (faithfulness) et la pertinence de la réponse générée par Mistral par rapport au contexte fourni. La réponse doit être bien rédigée, concise et répondre directement à la question de l'utilisateur.

Latence (Temps de Réponse) : S'assurer que le système complet (API + RAG) offre un temps de réponse acceptable pour une expérience utilisateur fluide (cible typique : quelques secondes ou moins).


- Périmètre : Zone géographique ciblée, période d’événements, données utilisées.

## Architecture du système
### Schéma global :

![Schéma d'architecture](./Architecture.png)

### Données entrantes (API Open Agenda)

**Source de données :** API Open Agenda v2 (https://api.openagenda.com/v2)

**Endpoints utilisés :**
- `/agendas` : Récupération des agendas culturels officiels par région
- `/agendas/{uid}/events` : Récupération des événements pour chaque agenda

**Paramètres de collecte :**
- **Région ciblée** : Occitanie (configurable via `OA_REGION` dans .env)
- **Pagination** : Curseur `after[]` avec taille de page de 100 événements (`OA_PAGE_SIZE=100`)
- **Filtrage temporel** :
  - Agendas : `updatedAt >= date` (dernière exécution ou 1 an par défaut)
  - Événements : `createdAt >= date` OU `updatedAt >= date` (mode UPDATE)

**Données extraites par événement :**
- **Métadonnées** : uid, title, description, slug
- **Temporalité** : timings (date_debut, date_fin), createdAt, updatedAt
- **Localisation** : location (coordinates, name, address, city, region)
- **Classification** : keywords, categories
- **Relations** : agendaUid (lien avec l'agenda parent)

**Stockage intermédiaire :**
- **Base MongoDB** : Collections `agendas` et `events`
- **Stratégie upsert** : Évite les doublons grâce à des clés uniques (uid pour agendas, (uid, agendaUid) pour events)
- **Dédoublonnement** : Script de nettoyage pour éliminer les événements dupliqués par uid

**Mise à jour incrémentale :**
- Pipeline de mise à jour qui sauvegarde les collections existantes (`_update_YYYYMMDD_HHMMSS`)
- Récupération sélective des agendas/événements modifiés depuis la dernière exécution
- Tracking des exécutions dans la collection `last_update` avec métadonnées complètes

### Prétraitement / embeddings / base vectorielle

**Pipeline de traitement (src/chunks/chunks_document.py) :**

1. **Formatage des documents**
   - Conversion des événements MongoDB en texte structuré
   - Format : `Titre: {title}\nDates: {date_debut} - {date_fin}\nDescription: {description}\nLieu: {locationName}\nMots-clés: {keywords}`

2. **Extraction des métadonnées**
   - Champs conservés : event_id, title, city, date_debut, date_fin, location (coordonnées GPS), region, keywords

3. **Chunking (LangChain RecursiveCharacterTextSplitter)**
   - **Taille des chunks** : 500 caractères (configurable via `CHUNK_SIZE`)
   - **Overlap** : 100 caractères (configurable via `CHUNK_OVERLAP`)
   - **Raison** : Équilibre entre contexte suffisant et précision de la recherche
   - **Sortie** : Objets LangChain `Document` avec contenu + métadonnées

**Génération des embeddings (src/embeddings/embeddings.py) :**

- **Modèle** : `intfloat/multilingual-e5-large` (HuggingFace Transformers)
- **Dimensionnalité** : 1024 dimensions
- **Multilingue** : Support de 100+ langues incluant le français
- **Local** : Pas d'API externe, inférence locale (pas de clé API requise)
- **Stratégie** :
  - Average pooling avec masque d'attention
  - Préfixes : "passage:" pour documents, "query:" pour requêtes
  - Normalisation L2 pour similarité cosinus optimale
- **Performance** :
  - Détection automatique du device (CUDA, MPS, CPU)
  - Batch processing (taille de batch configurable, défaut : 32)
  - ~50-100 chunks/seconde sur Apple Silicon (MPS)

**Construction de la base vectorielle (src/vectors/vectors.py) :**

- **Bibliothèque** : FAISS (Facebook AI Similarity Search)
- **Type d'index** : FAISS avec LangChain wrapper
- **Persistance** :
  - Format : Fichiers binaires FAISS + pickle pour métadonnées
  - Chemin : `data/faiss_index/` (configurable via `FAISS_INDEX_PATH`)
  - Sauvegarde : `index.faiss` + `index.pkl`
- **Métadonnées stockées** : Toutes les métadonnées extraites sont conservées avec chaque vecteur
- **Opérations supportées** :
  - Création d'index
  - Chargement d'index existant
  - Ajout de documents
  - Recherche par similarité (similarity_search_with_score)
  - Suppression d'index
  - Statistiques (nombre de vecteurs, dimension)

**Statistiques actuelles :**
- ~28,962 vecteurs indexés (exemple du développement)
- Dimension : 1024
- Couvre tous les événements culturels d'Occitanie récents

### Intégration LLM avec LangChain

**Modèle LLM sélectionné :**
- **Fournisseur** : Mistral AI
- **Modèle** : `mistral-small-latest` (configurable via `MISTRAL_MODEL`)
- **Raisons du choix** :
  - Excellence sur le français
  - Rapport qualité/coût optimal
  - Latence faible
  - Compatibilité native avec LangChain
  - API simple et fiable

**Architecture RAG (Retrieval-Augmented Generation) :**

1. **Recherche sémantique (Retrieval)**
   - Query embedding avec le même modèle E5
   - Recherche FAISS des top-k documents similaires (k=5 par défaut)
   - Récupération du contenu + métadonnées + scores de similarité

2. **Enrichissement du contexte**
   - Formatage des documents récupérés en contexte structuré
   - Inclusion des métadonnées pertinentes (titre, lieu, dates)
   - Limitation du contexte pour éviter le dépassement de tokens

3. **Génération de réponse**
   - **Prompt système** : Chargé depuis `src/chat/ps.md` (Puls-Events persona)
   - **Directives** :
     - Réponses basées uniquement sur le contexte fourni
     - Champ d'application : Occitanie et événements culturels
     - Ton : Enthousiaste, accueillant, clair et concis
     - Gestion de l'ambiguïté et des questions hors-sujet
   - **Prompt utilisateur enrichi** : Question + contexte RAG
   - **Appel Mistral AI** : Via `mistral_client.chat.complete()`
   - **Retour** : Réponse + contexte utilisé + statistiques tokens

**Implémentation LangChain :**
- **Custom Embeddings** : Classe `E5Embeddings(Embeddings)` compatible LangChain
- **Vector Store** : Wrapper FAISS de LangChain
- **Retrieval** : `vector_store.similarity_search_with_score(query, k=k)`
- **Messages** : `SystemMessage` + `UserMessage` pour Mistral AI

**Gestion de la qualité :**
- Système de scoring de similarité pour filtrer les résultats peu pertinents
- Limitation du nombre de documents contextuels (évite la surcharge)
- Tracking des tokens utilisés (prompt + completion + total)
- Fallback gracieux si pas de contexte pertinent trouvé

### Exposition via API

**Framework** : FastAPI 0.120.1+

**Architecture de l'API (src/api/main.py) :**

**Endpoints principaux :**

1. **GET /** - Point d'entrée
   - Liste tous les endpoints disponibles
   - Version de l'API

2. **GET /health** - Health check
   - Statut : ok | degraded
   - État des composants : vector_store, embeddings_model, mistral_client
   - Permet le monitoring

3. **GET /stats** - Statistiques du vector store
   - Nombre de vecteurs indexés
   - Dimension des vecteurs
   - Chemin de l'index

4. **POST /search** - Recherche sémantique pure
   - **Entrée** : `{"query": "...", "k": 5}`
   - **Sortie** : Résultats avec scores, titres, contenus, métadonnées
   - **Validation** : query non vide, k entre 1 et 100

5. **POST /ask** - Question-réponse avec RAG + Mistral AI
   - **Entrée** : `{"question": "...", "k": 5, "system_prompt": "..." (optionnel)}`
   - **Processus** :
     1. Recherche sémantique (top-k documents)
     2. Enrichissement du prompt avec contexte
     3. Appel Mistral AI
     4. Retour de la réponse
   - **Sortie** : `{"question": "...", "answer": "...", "context_used": [...], "tokens_used": {...}}`

6. **POST /rebuild** - Reconstruction incrémentale de l'index FAISS
   - Lance `update_pipeline.py` en arrière-plan
   - Vérification préalable : nouveaux événements présents ?
   - **Workflow** :
     1. Récupère la date de dernière exécution
     2. Compte les nouveaux événements MongoDB
     3. Si aucun : annule avec statut "warning"
     4. Sinon : lance le pipeline complet
     5. Recharge automatiquement l'index en mémoire
   - **Statuts** : started | running | success | success_with_warning | warning | error
   - Protection anti-concurrence (un seul rebuild à la fois)

7. **GET /rebuild/status** - Suivi du rebuild
   - Statut actuel et détails
   - Date de dernière mise à jour
   - Timestamps de démarrage

**Fonctionnalités techniques :**

- **CORS** : Configuré pour autoriser les requêtes cross-origin
- **Startup event** : Chargement automatique du vector store + embeddings + Mistral client au démarrage
- **Background tasks** : Exécution asynchrone du rebuild sans bloquer l'API
- **Rechargement automatique** : Nouvel index FAISS chargé en mémoire après rebuild réussi
- **Auto-documentation** : Swagger UI accessible à `/docs`
- **Gestion d'erreurs** : HTTPException avec codes appropriés (422, 503, 500)
- **Logging** : Logs détaillés de toutes les opérations
- **Hot-reload** : Activé en mode développement

**Format des réponses :**
- JSON structuré avec modèles Pydantic (validation automatique)
- Codes HTTP standards
- Messages d'erreur explicites

**Déploiement :**
- **Développement** : `uvicorn` avec hot-reload (`make run-api`)
- **Production** : Docker + docker-compose
- **Port** : 8000 (configurable)
- **Host** : 0.0.0.0

**Tests unitaires :**
- 10 tests passants couvrant tous les endpoints principaux
- Mocking complet des dépendances (MongoDB, FAISS, Mistral)
- Pytest avec support async
- Commandes : `make test` | `make test-cov`

### Technologies utilisées :

**Backend et orchestration :**
- **Python 3.13+** : Langage principal
- **FastAPI 0.120.1+** : Framework web moderne et rapide
- **Uvicorn** : Serveur ASGI avec hot-reload
- **LangChain 1.0.2+** : Framework d'orchestration LLM
  - `langchain-community` : Intégrations communautaires
  - `langchain-mistralai` : Connecteur Mistral AI
  - `langchain-text-splitters` : Chunking de documents
- **python-dotenv** : Gestion des variables d'environnement

**LLM et embeddings :**
- **Mistral AI API (mistralai 1.9.11+)** : Génération de réponses
  - Modèle : `mistral-small-latest`
  - SystemMessage/UserMessage pour le chat
- **HuggingFace Transformers 4.57.1+** : Modèles NLP locaux
  - `intfloat/multilingual-e5-large` : Embeddings multilingues (1024 dim)
- **PyTorch 2.9.0+** : Backend pour les transformers
  - Support CUDA/MPS/CPU

**Base de données et vectorielle :**
- **MongoDB (PyMongo 4.15.3+ / Motor 3.7.1+)** : Stockage des événements
  - Collections : `agendas`, `events`, `last_update`
  - Opérations bulk avec upsert
- **FAISS (faiss-cpu 1.12.0+)** : Recherche vectorielle
  - Index persistant sur disque
  - Recherche par similarité cosinus

**Scraping et API :**
- **Requests 2.32.5+** : Appels HTTP vers Open Agenda API
- **BeautifulSoup4 (bs4 0.0.2+)** : Parsing HTML si nécessaire

**Validation et modèles de données :**
- **Pydantic 2.12.3+** : Validation des données API
  - Modèles pour requêtes/réponses FastAPI

**Outils de développement :**
- **pytest 8.3.0+** : Tests unitaires
  - `pytest-asyncio 0.24.0+` : Support async
  - `pytest-cov 6.0.0+` : Couverture de code
  - `httpx 0.28.0+` : Client HTTP async pour tests
- **flake8 7.3.0+** : Linter Python
- **uv** : Gestionnaire de dépendances rapide
- **Make** : Orchestration des commandes

**Conteneurisation et déploiement :**
- **Docker** : Conteneurisation de l'application
- **docker-compose** : Orchestration multi-conteneurs
  - Service MongoDB
  - Service API FastAPI

**Utilitaires :**
- **NumPy 2.3.4+** : Calculs matriciels
- **pathlib** : Manipulation de chemins
- **asyncio** : Programmation asynchrone
- **logging** : Journalisation applicative

**Configuration système :**
- **macOS fix** : `KMP_DUPLICATE_LIB_OK=TRUE` pour OpenMP
- **Environnement** : Fichier `.env` pour la configuration
- **Makefile** : Commandes standardisées (`make install`, `make run-all`, etc.)


3. Préparation et vectorisation des données
Source de données : API Open Agenda (paramètres utilisés, filtres appliqués)
Nettoyage : Exemples d’anomalies corrigées, méthodes utilisées
Chunking : Raison du découpage, taille choisie
Embedding :
Modèle utilisé (ex. : Mistral embedding API)
Dimensionnalité, logique de batch, format des vecteurs

4. Choix du modèle NLP
Modèle sélectionné :
Pourquoi ce modèle ? (Critères : coût, qualité, compatibilité LangChain…)
Prompting (si utilisé) : Prompt de base / structure
Limites du modèle : 

5. Construction de la base vectorielle
Faiss utilisé :
Stratégie de persistance : 
Format de sauvegarde ?
nommage ?
Métadonnées associées : 
Ce qui est conservé pour chaque document

6. API et endpoints exposés
Framework utilisé : FastAPI / Flask
Endpoints clés :
/ask : question utilisateur, réponse du système
/rebuild : reconstruction de l’index (si besoin)
Format des requêtes/réponses
Exemple d’appel API : avec curl ou code Python
Tests effectués et documentés
Gestion des erreurs / limitations

7. Évaluation du système
Jeu de test annoté :
Nombre d’exemples
Méthode d’annotation
Métriques d’évaluation :
Exemple : similarité sémantique, taux de couverture des réponses, score de satisfaction (subjectif)
Résultats obtenus :
Analyse quantitative (scores globaux)
Analyse qualitative (exemples de bonnes/mauvaises réponses)

8. Recommandations et perspectives
Ce qui fonctionne bien
Limites du POC :
Volumétrie, performance, coût, couverture thématique ?
Améliorations possibles :
Ajout de…Amélioration de..
Passage en production via…

9.  Organisation du dépôt GitHub
Arborescence du dépôt (fichiers clés, scripts, dossiers)
Explication rapide de chaque répertoire

10. Annexes (exemples)
Extraits du jeu de test annoté
Prompt utilisé (si spécifique)
Extraits de logs ou exemples de réponse JSON
