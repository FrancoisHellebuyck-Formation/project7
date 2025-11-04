"""
API FastAPI pour l'interrogation du vector store FAISS.

Cette API permet d'effectuer des recherches sémantiques sur les événements culturels
en utilisant le vector store FAISS pré-calculé.
"""

import logging
import os
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral, UserMessage, SystemMessage

from embeddings.embeddings import get_embeddings_model
from vectors.vectors import load_vector_store, get_vector_store_stats
from api.models import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    AskQuery,
    AskResponse,
    StatsResponse,
    HealthResponse,
    RebuildResponse,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Configuration
# Note: Le chemin doit être absolu ou relatif au répertoire racine du projet
_faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
# Si le chemin n'est pas absolu, le rendre relatif au répertoire racine du projet
# __file__ est src/api/main.py, donc on remonte 3 niveaux pour arriver à la racine
if not os.path.isabs(_faiss_index_path):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    FAISS_INDEX_PATH = os.path.join(project_root, _faiss_index_path)
else:
    FAISS_INDEX_PATH = _faiss_index_path

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
EMBEDDINGS_DEVICE = os.getenv("EMBEDDINGS_DEVICE") or None

# Configuration Mistral AI
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de recherche d'événements culturels",
    description="API pour effectuer des recherches sémantiques sur les événements culturels de la région Occitanie",
    version="1.0.0",
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le vector store et le modèle d'embeddings
vector_store = None
embeddings_model = None
mistral_client = None
default_system_prompt = None

# Variables pour suivre l'état du rebuild
rebuild_in_progress = False
rebuild_status = {
    "status": "idle",
    "message": "Aucun rebuild en cours",
    "started_at": None,
    "last_update_date": None
}


def load_system_prompt(file_path: str) -> str:
    """
    Charge le prompt système depuis un fichier markdown.

    Args:
        file_path: Chemin vers le fichier .md contenant le prompt système

    Returns:
        Contenu du fichier comme chaîne de caractères

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"✓ Prompt système chargé depuis: {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"❌ Fichier de prompt système introuvable: {file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du prompt système: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialise le vector store et le modèle d'embeddings au démarrage."""
    global vector_store, embeddings_model, mistral_client, default_system_prompt

    logger.info("=" * 70)
    logger.info("DÉMARRAGE DE L'API DE RECHERCHE")
    logger.info("=" * 70)

    try:
        # Chargement du modèle d'embeddings
        logger.info("Chargement du modèle d'embeddings...")
        embeddings_model = get_embeddings_model(
            model_id=EMBEDDINGS_MODEL,
            device=EMBEDDINGS_DEVICE
        )
        logger.info("✓ Modèle d'embeddings chargé")

        # Chargement du vector store
        logger.info(f"Chargement du vector store depuis: {FAISS_INDEX_PATH}")
        vector_store = load_vector_store(
            load_path=FAISS_INDEX_PATH,
            embeddings=embeddings_model
        )

        # Affichage des statistiques
        stats = get_vector_store_stats(vector_store)
        logger.info("✓ Vector store chargé")
        logger.info(f"  - Nombre de vecteurs: {stats['num_vectors']:,}")
        logger.info(f"  - Dimension: {stats['dimension']}")

        # Initialisation du client Mistral AI (si clé API disponible)
        if MISTRAL_API_KEY:
            logger.info("Initialisation du client Mistral AI...")
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            logger.info("✓ Client Mistral AI initialisé")

            # Chargement du prompt système depuis le fichier ps.md
            # Le fichier ps.md est dans src/chat/, et ce fichier est src/api/main.py
            # Donc on remonte d'un niveau puis on va dans chat/
            prompt_file_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "chat",
                "ps.md"
            )
            logger.info(f"Chargement du prompt système depuis: {prompt_file_path}")
            default_system_prompt = load_system_prompt(prompt_file_path)
        else:
            logger.warning("⚠️  MISTRAL_API_KEY non configurée - endpoint /ask désactivé")

        logger.info("=" * 70)
        logger.info("✓ API PRÊTE À RECEVOIR DES REQUÊTES")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Point d'entrée racine de l'API."""
    return {
        "message": "API de recherche d'événements culturels",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "ask": "/ask",
            "stats": "/stats",
            "health": "/health",
            "rebuild": "/rebuild",
            "rebuild_status": "/rebuild/status",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé de l'API."""
    return HealthResponse(
        status="ok" if vector_store and embeddings_model else "degraded",
        vector_store_loaded=vector_store is not None,
        embeddings_model_loaded=embeddings_model is not None,
        mistral_client_loaded=mistral_client is not None
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Retourne les statistiques du vector store."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store non chargé")

    try:
        stats = get_vector_store_stats(vector_store)
        return StatsResponse(
            num_vectors=stats["num_vectors"],
            dimension=stats["dimension"],
            index_path=FAISS_INDEX_PATH
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Effectue une recherche sémantique sur les événements culturels (méthode POST).

    Args:
        query: Objet contenant la requête et le nombre de résultats souhaités

    Returns:
        Liste des résultats de recherche avec scores et métadonnées
    """
    if not vector_store or not embeddings_model:
        raise HTTPException(status_code=503, detail="Vector store ou modèle d'embeddings non chargé")

    try:
        logger.info(f"Recherche: '{query.query}' (k={query.k})")

        # Recherche dans le vector store
        results = vector_store.similarity_search_with_score(query.query, k=query.k)

        # Formatage des résultats
        formatted_results = []
        for doc, score in results:
            result = SearchResult(
                score=float(score),
                title=doc.metadata.get("title", "Sans titre"),
                content=doc.page_content,
                location=doc.metadata.get("location"),
                metadata=doc.metadata
            )
            formatted_results.append(result)

        logger.info(f"✓ {len(formatted_results)} résultats trouvés")

        return SearchResponse(
            query=query.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(query: AskQuery):
    """
    Répond à une question en utilisant RAG + Mistral AI.

    Cette endpoint combine la recherche sémantique (RAG) avec l'API Mistral AI
    pour fournir des réponses contextuelles basées sur les événements culturels.

    Workflow:
    1. Recherche sémantique dans le vector store (top-k résultats)
    2. Formatage du contexte avec les événements trouvés
    3. Enrichissement du prompt utilisateur
    4. Appel à Mistral AI pour générer la réponse
    5. Retour de la réponse avec contexte et statistiques

    Args:
        query: Objet contenant la question et les paramètres

    Returns:
        Réponse générée avec contexte et statistiques d'utilisation
    """
    if not vector_store or not embeddings_model:
        raise HTTPException(
            status_code=503,
            detail="Vector store ou modèle d'embeddings non chargé"
        )

    if not mistral_client:
        raise HTTPException(
            status_code=503,
            detail="Client Mistral AI non initialisé. Vérifiez MISTRAL_API_KEY dans .env"
        )

    try:
        logger.info(f"Question reçue: '{query.question}' (k={query.k})")

        # 1. Recherche sémantique dans le vector store
        logger.info(f"Recherche de {query.k} documents contextuels...")
        results = vector_store.similarity_search_with_score(query.question, k=query.k)

        # 2. Formatage du contexte
        context_results = []
        context_parts = ["Voici les informations pertinentes trouvées dans la base de données:\n"]

        for i, (doc, score) in enumerate(results, 1):
            # Créer le SearchResult pour la réponse
            search_result = SearchResult(
                score=float(score),
                title=doc.metadata.get("title", "Sans titre"),
                content=doc.page_content,
                location=doc.metadata.get("location"),
                metadata=doc.metadata
            )
            context_results.append(search_result)

            # Formater pour le contexte textuel
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content

            context_parts.append(f"\n--- Résultat {i} (pertinence: {score:.3f}) ---")
            context_parts.append(f"Titre: {doc.metadata.get('title', 'Sans titre')}")

            if doc.metadata.get("city"):
                context_parts.append(f"Ville: {doc.metadata['city']}")
            if doc.metadata.get("date_debut"):
                context_parts.append(f"Date début: {doc.metadata['date_debut']}")
            if doc.metadata.get("date_fin"):
                context_parts.append(f"Date fin: {doc.metadata['date_fin']}")

            context_parts.append(f"\nContenu:\n{content_preview}")

        rag_context = "\n".join(context_parts)
        logger.info(f"✓ {len(context_results)} documents trouvés pour le contexte")

        # 3. Construction du prompt enrichi
        enriched_prompt = f"""{rag_context}

---

Question de l'utilisateur:
{query.question}

Réponds à la question en te basant sur les informations contextuelles ci-dessus. Si les informations ne permettent pas de répondre complètement, indique-le clairement."""

        # 4. Préparation des messages pour Mistral AI
        # Utilise le prompt système personnalisé si fourni, sinon utilise le prompt par défaut chargé depuis ps.md
        system_prompt = query.system_prompt or default_system_prompt

        if not system_prompt:
            # Fallback en cas de problème de chargement du fichier ps.md
            logger.warning("⚠️  Aucun prompt système disponible, utilisation d'un prompt par défaut minimal")
            system_prompt = """Tu es un assistant spécialisé dans les événements culturels de la région Occitanie.
Tu dois répondre aux questions des utilisateurs en te basant UNIQUEMENT sur les informations fournies dans le contexte.
Si tu ne trouves pas l'information dans le contexte, dis-le clairement.
Sois précis, concis et utile."""

        messages = [
            SystemMessage(content=system_prompt, role="system"),
            UserMessage(content=enriched_prompt, role="user")
        ]

        # 5. Appel à Mistral AI
        logger.info(f"Appel à Mistral AI (modèle: {MISTRAL_MODEL})...")
        response = mistral_client.chat.complete(model=MISTRAL_MODEL, messages=messages)

        # 6. Extraction de la réponse
        answer = response.choices[0].message.content

        # 7. Statistiques d'utilisation
        tokens_stats = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        logger.info(f"✓ Réponse générée (tokens: {tokens_stats['total_tokens']})")

        return AskResponse(
            question=query.question,
            answer=answer,
            context_used=context_results,
            tokens_used=tokens_stats
        )

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def run_rebuild_pipeline():
    """
    Exécute le pipeline de mise à jour incrémentale en arrière-plan.

    Cette fonction lance le script update_pipeline.py qui effectue:
    1. Récupération de la date de dernière exécution
    2. Backup et vidage des collections MongoDB
    3. Récupération des agendas mis à jour
    4. Récupération des événements
    5. Dédoublonnement
    6. Chunking et génération des embeddings
    7. Mise à jour de l'index FAISS
    """
    global rebuild_in_progress

    try:
        rebuild_status["status"] = "running"
        rebuild_status["message"] = "Pipeline de mise à jour en cours..."
        rebuild_status["started_at"] = datetime.now().isoformat()

        logger.info("=" * 70)
        logger.info("🔄 DÉMARRAGE DU REBUILD DE L'INDEX FAISS")
        logger.info("=" * 70)

        # Récupérer la date de dernière mise à jour
        from pymongo import MongoClient
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGODB_DB_NAME", "OA")

        client = None
        last_update_date = None
        try:
            client = MongoClient(mongodb_uri)
            db = client[db_name]
            last_update_collection = db["last_update"]

            last_execution = last_update_collection.find_one(
                {}, sort=[("pipeline_run_date", -1)]
            )

            if last_execution and "pipeline_run_date" in last_execution:
                run_date = last_execution["pipeline_run_date"]
                if isinstance(run_date, datetime):
                    last_update_date = run_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                else:
                    last_update_date = str(run_date)

                rebuild_status["last_update_date"] = last_update_date
                logger.info(f"✓ Date de dernière exécution: {last_update_date}")

            # Vérifier s'il y a de nouveaux événements depuis la dernière exécution
            if last_update_date:
                events_collection = db[
                    os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")
                ]

                # Compter les événements créés ou mis à jour depuis la dernière exécution
                new_events_count = events_collection.count_documents({
                    "$or": [
                        {"createdAt": {"$gte": last_update_date}},
                        {"updatedAt": {"$gte": last_update_date}}
                    ]
                })

                logger.info(
                    f"📊 Événements nouveaux/modifiés depuis la dernière "
                    f"exécution: {new_events_count}"
                )

                if new_events_count == 0:
                    logger.warning("⚠️  Aucun nouvel événement détecté")
                    rebuild_status["status"] = "warning"
                    rebuild_status["message"] = (
                        "Pas de nouveaux événements depuis la dernière exécution. "
                        "Rebuild annulé."
                    )
                    rebuild_in_progress = False
                    return

        finally:
            if client:
                client.close()

        # Construire le chemin vers le script update_pipeline.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        script_path = os.path.join(project_root, "src", "update_pipeline.py")

        # Exécuter le pipeline de mise à jour
        logger.info(f"Exécution du script: {script_path}")
        process = await asyncio.create_subprocess_exec(
            "uv", "run", "python", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_root
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info("✅ Pipeline de mise à jour terminé avec succès")
            logger.info("🔄 Rechargement de l'index FAISS en mémoire...")

            # Recharger le vector store avec le nouvel index
            try:
                global vector_store
                vector_store = load_vector_store(
                    load_path=FAISS_INDEX_PATH,
                    embeddings=embeddings_model,
                    verbose=False
                )

                # Afficher les nouvelles statistiques
                stats = get_vector_store_stats(vector_store)
                logger.info("✅ Nouvel index FAISS chargé en mémoire")
                logger.info(f"  - Nombre de vecteurs: {stats['num_vectors']:,}")
                logger.info(f"  - Dimension: {stats['dimension']}")

                rebuild_status["status"] = "success"
                rebuild_status["message"] = (
                    "Pipeline terminé avec succès. "
                    "Nouvel index FAISS chargé automatiquement."
                )
            except Exception as reload_error:
                logger.error(
                    f"❌ Erreur lors du rechargement de l'index: {reload_error}",
                    exc_info=True
                )
                rebuild_status["status"] = "success_with_warning"
                rebuild_status["message"] = (
                    "Pipeline terminé avec succès mais échec du rechargement. "
                    "Redémarrez l'API manuellement pour charger le nouvel index."
                )

        else:
            error_msg = stderr.decode() if stderr else "Erreur inconnue"
            rebuild_status["status"] = "error"
            rebuild_status["message"] = f"Échec du pipeline: {error_msg}"
            logger.error(f"❌ Échec du pipeline de mise à jour: {error_msg}")

    except Exception as e:
        rebuild_status["status"] = "error"
        rebuild_status["message"] = f"Erreur lors du rebuild: {str(e)}"
        logger.error(f"❌ Erreur lors du rebuild: {e}", exc_info=True)
    finally:
        rebuild_in_progress = False


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Lance le pipeline de mise à jour incrémentale de l'index FAISS.

    Cette endpoint déclenche le pipeline de mise à jour qui:
    1. Récupère la date de dernière exécution
    2. Sauvegarde et vide les collections MongoDB
    3. Récupère les agendas mis à jour depuis la dernière exécution
    4. Récupère les événements pour ces agendas (avec filtre de date)
    5. Dédoublonne les événements
    6. Génère les chunks et les embeddings
    7. Reconstruit l'index FAISS

    Le pipeline s'exécute en arrière-plan. Utilisez GET /rebuild/status
    pour suivre la progression.

    IMPORTANT: Une fois le rebuild terminé, vous devez redémarrer l'API
    pour charger le nouvel index FAISS en mémoire.

    Returns:
        RebuildResponse avec le statut de l'opération
    """
    global rebuild_in_progress

    if rebuild_in_progress:
        return RebuildResponse(
            status="running",
            message="Un rebuild est déjà en cours",
            last_update_date=rebuild_status.get("last_update_date"),
            details={
                "started_at": rebuild_status.get("started_at"),
                "current_status": rebuild_status.get("message")
            }
        )

    rebuild_in_progress = True
    background_tasks.add_task(run_rebuild_pipeline)

    return RebuildResponse(
        status="started",
        message=(
            "Pipeline de mise à jour démarré en arrière-plan. "
            "Utilisez GET /rebuild/status pour suivre la progression."
        ),
        last_update_date=None,
        details={"started_at": datetime.now().isoformat()}
    )


@app.get("/rebuild/status", response_model=RebuildResponse)
async def rebuild_status_endpoint():
    """
    Retourne le statut du rebuild en cours ou du dernier rebuild.

    Returns:
        RebuildResponse avec le statut actuel
    """
    return RebuildResponse(
        status=rebuild_status["status"],
        message=rebuild_status["message"],
        last_update_date=rebuild_status.get("last_update_date"),
        details={
            "started_at": rebuild_status.get("started_at"),
            "in_progress": rebuild_in_progress
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Démarrage du serveur en mode développement
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
