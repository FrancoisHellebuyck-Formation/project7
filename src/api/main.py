"""
API FastAPI pour l'interrogation du vector store FAISS.

Cette API permet d'effectuer des recherches sémantiques sur les événements culturels
en utilisant le vector store FAISS pré-calculé.
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from embeddings.embeddings import get_embeddings_model
from vectors.vectors import load_vector_store, get_vector_store_stats

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


# Modèles Pydantic pour les requêtes et réponses
class SearchQuery(BaseModel):
    """Modèle pour une requête de recherche."""
    query: str = Field(..., description="Texte de la requête de recherche", min_length=1)
    k: int = Field(5, description="Nombre de résultats à retourner", ge=1, le=100)


class SearchResult(BaseModel):
    """Modèle pour un résultat de recherche."""
    score: float = Field(..., description="Score de similarité (distance L2)")
    title: str = Field(..., description="Titre de l'événement")
    content: str = Field(..., description="Contenu du document")
    location: Optional[str] = Field(None, description="Lieu de l'événement")
    metadata: dict = Field(default_factory=dict, description="Métadonnées supplémentaires")


class SearchResponse(BaseModel):
    """Modèle pour la réponse de recherche."""
    query: str = Field(..., description="Requête effectuée")
    results: List[SearchResult] = Field(..., description="Liste des résultats")
    total_results: int = Field(..., description="Nombre de résultats retournés")


class StatsResponse(BaseModel):
    """Modèle pour les statistiques du vector store."""
    num_vectors: int = Field(..., description="Nombre de vecteurs dans l'index")
    dimension: int = Field(..., description="Dimension des vecteurs")
    index_path: str = Field(..., description="Chemin du vector store")


class HealthResponse(BaseModel):
    """Modèle pour le health check."""
    status: str = Field(..., description="Statut de l'API")
    vector_store_loaded: bool = Field(..., description="Indique si le vector store est chargé")
    embeddings_model_loaded: bool = Field(..., description="Indique si le modèle d'embeddings est chargé")


@app.on_event("startup")
async def startup_event():
    """Initialise le vector store et le modèle d'embeddings au démarrage."""
    global vector_store, embeddings_model

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
        embeddings_model_loaded=embeddings_model is not None
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


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Texte de la requête de recherche", min_length=1),
    k: int = Query(5, description="Nombre de résultats à retourner", ge=1, le=100)
):
    """
    Effectue une recherche sémantique sur les événements culturels.

    Args:
        q: Texte de la requête de recherche
        k: Nombre de résultats à retourner (par défaut: 5, max: 100)

    Returns:
        Liste des résultats de recherche avec scores et métadonnées
    """
    if not vector_store or not embeddings_model:
        raise HTTPException(status_code=503, detail="Vector store ou modèle d'embeddings non chargé")

    try:
        logger.info(f"Recherche: '{q}' (k={k})")

        # Recherche dans le vector store
        results = vector_store.similarity_search_with_score(q, k=k)

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
            query=q,
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_post(query: SearchQuery):
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
