"""
API FastAPI pour l'interrogation du vector store FAISS.

Cette API permet d'effectuer des recherches sémantiques sur les événements culturels
en utilisant le vector store FAISS pré-calculé.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
