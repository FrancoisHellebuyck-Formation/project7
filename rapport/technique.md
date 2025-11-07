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


### Préparation et vectorisation des données
#### Source de données : API Open Agenda (paramètres utilisés, filtres appliqués)
- **Endpoints** : `/agendas` pour lister les agendas officiels, puis `/agendas/{uid}/events` pour récupérer les événements.
- **Paramètres clés** : `official: 1` pour ne retenir que les sources fiables, `search: Occitanie` pour le ciblage géographique.
- **Filtres temporels** : Le pipeline de mise à jour incrémentale (`update`) filtre les agendas et événements sur la base de la date de la dernière exécution (`createdAt >= date` ou `updatedAt >= date`), assurant une collecte efficace des nouveautés.

#### Nettoyage : Exemples d’anomalies corrigées, méthodes utilisées
- **Anomalie corrigée** : Présence d'événements en double, identifiés par un `uid` identique mais des `_id` MongoDB différents.
- **Méthode utilisée** : Le script `src/corpus/deduplicate_events.py` est exécuté après la collecte. Pour chaque `uid` dupliqué, il conserve uniquement l'événement le plus récent en se basant sur le champ `updatedAt` et supprime les autres.

#### Chunking : Raison du découpage, taille choisie
- **Outil** : `RecursiveCharacterTextSplitter` de LangChain.
- **Taille choisie** : **500 caractères** (`CHUNK_SIZE`) avec un chevauchement de **100 caractères** (`CHUNK_OVERLAP`).
- **Raison du découpage** : Cette configuration offre un équilibre optimal. Les chunks sont assez petits pour que la recherche sémantique soit très précise, mais assez grands pour conserver un contexte sémantique suffisant. Le chevauchement empêche de couper des phrases ou des idées importantes entre deux chunks.

#### Embedding :
##### Modèle utilisé (ex. : Mistral embedding API)
- **Modèle** : `intfloat/multilingual-e5-large`, un modèle de pointe exécuté localement via la bibliothèque HuggingFace Transformers.
- **Justification** : Ce choix a été fait pour ses excellentes performances sur les tâches de *retrieval* en français, sa capacité à s'exécuter localement (pas de dépendance à une API externe, pas de coût par token) et sa dimensionnalité élevée.

##### Dimensionnalité, logique de batch, format des vecteurs
- **Dimensionnalité** : **1024 dimensions**, ce qui permet une représentation sémantique très riche.
- **Logique de batch** : La vectorisation est effectuée par lots (taille de 32 par défaut) pour optimiser l'utilisation des ressources matérielles (CPU/GPU/MPS) et accélérer le traitement.
- **Format des vecteurs** : Les vecteurs sont des flottants normalisés (L2), ce qui est idéal pour les calculs de similarité cosinus. Le modèle utilise des préfixes spécifiques (`"passage:"` pour les documents, `"query:"` pour les requêtes) afin d'améliorer la pertinence de la recherche.

### Choix du modèle NLP
##### Modèle sélectionné :
- **Fournisseur** : Mistral AI
- **Modèle** : `mistral-small-latest` (configurable via la variable d'environnement `MISTRAL_MODEL`)

##### Pourquoi ce modèle ? (Critères : coût, qualité, compatibilité LangChain…)
- **Qualité sur le français** : Les modèles Mistral sont reconnus pour leur excellente performance et leur compréhension nuancée de la langue française.
- **Rapport performance/coût** : `mistral-small-latest` offre un excellent équilibre entre une latence faible, une haute qualité de génération et un coût par token maîtrisé, ce qui est idéal pour un POC.
- **Compatibilité LangChain** : Le modèle est nativement supporté via le package `langchain-mistralai`, permettant une intégration simple et rapide dans l'architecture RAG.

##### Prompting (si utilisé) : Prompt de base / structure
- **Prompt Système** : Un prompt système détaillé, stocké dans `src/chat/ps.md`, définit la personnalité du chatbot ("Puls-Events"). Il lui donne des instructions strictes : répondre uniquement sur la base du contexte fourni, se limiter à la région Occitanie, et adopter un ton convivial et précis.
- **Structure du prompt enrichi** : La requête finale envoyée à Mistral est structurée en deux parties :
    1.  `SystemMessage` : Contient les instructions de `ps.md`.
    2.  `UserMessage` : Contient un prompt enrichi qui combine :
        - Le contexte récupéré depuis la base vectorielle (les événements pertinents).
        - La question originale de l'utilisateur.
        - Une instruction finale demandant de baser la réponse sur le contexte.

##### Limites du modèle :
- **Dépendance à une API externe** : Contrairement au modèle d'embedding, l'utilisation de Mistral AI nécessite une connexion internet et une clé API valide, ce qui engendre un coût par utilisation (basé sur les tokens).
- **Fenêtre de contexte** : Le modèle a une taille de contexte limitée. Le nombre de documents injectés dans le prompt (`RAG_TOP_K`) doit être contrôlé pour ne pas dépasser cette limite et pour maîtriser les coûts.

### Construction de la base vectorielle
##### Faiss utilisé :
- **Bibliothèque** : **FAISS** (Facebook AI Similarity Search), une bibliothèque hautement optimisée pour la recherche de similarité sur de grands volumes de vecteurs.
- **Intégration** : Le projet utilise le wrapper `FAISS` fourni par `langchain-community`, ce qui simplifie la création, la sauvegarde, le chargement et l'interrogation de l'index.

##### Stratégie de persistance :
- **Format de sauvegarde** : L'index est sauvegardé sur le disque dans le répertoire `data/faiss_index/` (configurable via `FAISS_INDEX_PATH`). Il se compose de deux fichiers :
    - `index.faiss` : Contient les vecteurs numériques dans un format binaire optimisé par FAISS.
    - `index.pkl` : Un fichier pickle contenant le mapping entre les index des vecteurs et les métadonnées des documents (`docstore`).

##### Métadonnées associées :
- Chaque chunk vectorisé conserve un ensemble riche de métadonnées extraites de l'événement original. Ces métadonnées sont cruciales pour filtrer, afficher et contextualiser les résultats de recherche.
- **Champs conservés** : `event_id`, `title`, `city`, `date_debut`, `date_fin`, `location` (coordonnées GPS), `region`, `keywords`.

### API et endpoints exposés
##### Framework utilisé : FastAPI
- L'API est développée avec **FastAPI**, un framework Python moderne et performant, et servie par **Uvicorn**, un serveur ASGI. Ce choix garantit des temps de réponse rapides et une scalabilité aisée.

##### Endpoints clés :
- **`POST /ask`** : Le cœur du système RAG. Prend une question en JSON, effectue une recherche sémantique pour trouver des contextes pertinents, enrichit un prompt et interroge le LLM (Mistral) pour générer une réponse factuelle.
- **`POST /search`** : Endpoint de recherche sémantique pure. Il retourne les `k` documents les plus pertinents de la base vectorielle avec leurs scores de similarité, sans passer par le LLM.
- **`POST /rebuild`** : Déclenche le pipeline de mise à jour incrémentale de l'index en arrière-plan. Il est non-bloquant et vérifie au préalable si de nouveaux événements justifient une mise à jour.
- **`GET /rebuild/status`** : Permet de suivre l'état d'avancement du pipeline de reconstruction (ex: `running`, `success`, `error`).
- **`GET /health`** et **`GET /stats`** : Endpoints de monitoring pour vérifier l'état de santé de l'API et les statistiques de l'index (nombre de vecteurs, etc.).

##### Format des requêtes/réponses
- Les formats sont validés par des modèles **Pydantic** pour assurer la robustesse.
- **Requête `/ask`** : `{"question": "...", "k": 5}`
- **Réponse `/ask`** :
  ```json
  {
    "question": "...",
    "answer": "...",
    "context_used": [ { "score": 0.8, "title": "...", ... } ],
    "tokens_used": { "prompt_tokens": ..., "total_tokens": ... }
  }
  ```

##### Exemple d’appel API : avec curl
- **Pour poser une question au RAG :**
  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "Quels sont les festivals de jazz en Occitanie ?", "k": 5}'
  ```
- **Pour lancer une reconstruction de l'index :**
  ```bash
  curl -X POST http://localhost:8000/rebuild
  ```

##### Tests effectués et documentés
- Le projet inclut une suite de tests unitaires complète utilisant **Pytest** et **HTTPX**.
- Les dépendances externes (FAISS, Mistral) sont **mockées** pour isoler les tests de l'API.
- Des tests d'évaluation de la qualité du RAG sont également implémentés avec **RAGAS** (`make test-ragas`) pour mesurer la pertinence et la fidélité des réponses.

##### Gestion des erreurs / limitations
- L'API utilise les `HTTPException` de FastAPI pour retourner des codes d'erreur standards (422 pour une requête invalide, 503 si un service est indisponible, 500 pour une erreur interne).
- L'endpoint `/rebuild` est protégé contre les exécutions concurrentes.

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
