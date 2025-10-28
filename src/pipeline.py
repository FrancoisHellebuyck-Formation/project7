"""
Pipeline complet pour la création de la base vectorielle depuis MongoDB.

Ce module orchestre l'ensemble du processus :
1. Connexion à MongoDB
2. Chargement et chunking des documents
3. Génération des embeddings
4. Création de l'index FAISS
5. Sauvegarde et test de recherche
"""

from typing import Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings_model
from vectors import create_vector_store, save_vector_store, search_similar_documents
from chunks.chunks_document import get_mongodb_connection, process_events_to_chunks

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_vector_store_pipeline(
    save_path: Optional[str] = None,
    mongodb_query: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    verbose: bool = False,
) -> FAISS:
    """
    Pipeline complet: MongoDB → chunks → embeddings → FAISS.

    Args:
        save_path: Chemin pour sauvegarder le vector store (optionnel)
        mongodb_query: Filtre MongoDB pour sélectionner les événements
        limit: Nombre maximum d'événements à traiter
        chunk_size: Taille des chunks en caractères
        chunk_overlap: Chevauchement entre chunks
        model_id: Identifiant du modèle d'embeddings
        device: Device à utiliser ('cuda', 'mps', 'cpu')
        batch_size: Taille des batchs pour les embeddings
        verbose: Si True, affiche des informations de progression

    Returns:
        FAISS: Instance du vector store créé

    Raises:
        ValueError: Si aucun chunk n'a pu être créé
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("PIPELINE DE CRÉATION DU VECTOR STORE")
        logger.info("=" * 70)

    # 1. Connexion à MongoDB
    if verbose:
        logger.info("\n[1/4] Connexion à MongoDB...")
    client, events_collection = get_mongodb_connection()

    try:
        # 2. Chargement et chunking des documents
        if verbose:
            logger.info("\n[2/4] Chargement et découpage des documents...")
        chunks = process_events_to_chunks(
            events_collection=events_collection,
            query=mongodb_query,
            limit=limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose,
        )

        if not chunks:
            raise ValueError(
                "Aucun chunk créé. Vérifiez que des événements existent dans MongoDB."
            )

        # 3. Création des embeddings et du vector store
        if verbose:
            logger.info(
                "\n[3/4] Génération des embeddings et création du vector store..."
            )
            logger.info(f"      Nombre de chunks: {len(chunks)}")
            logger.info(f"      Modèle: {model_id or 'intfloat/multilingual-e5-large'}")
            logger.info(f"      Device: {device or 'auto-détecté'}")
            logger.info(f"      Batch size: {batch_size}")
            logger.info("      Cette étape peut prendre plusieurs minutes...")

        embeddings = get_embeddings_model(
            model_id=model_id, device=device, batch_size=batch_size
        )
        vector_store = create_vector_store(chunks, embeddings, verbose=verbose)

        # 4. Sauvegarde du vector store
        if save_path:
            if verbose:
                logger.info("\n[4/4] Sauvegarde du vector store...")
            save_vector_store(vector_store, save_path, verbose=verbose)
        else:
            if verbose:
                logger.info("\n[4/4] Sauvegarde ignorée (aucun chemin spécifié)")

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("✓ PIPELINE TERMINÉ AVEC SUCCÈS")
            logger.info("=" * 70)

        return vector_store

    finally:
        client.close()
        if verbose:
            logger.info("Connexion MongoDB fermée")


def main():
    """
    Fonction principale pour exécuter le pipeline complet.
    Configuration via variables d'environnement.
    """
    # Chargement des variables d'environnement
    load_dotenv()

    # Configuration
    save_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    limit = os.getenv("EMBEDDINGS_LIMIT")
    limit = int(limit) if limit else None

    model_id = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
    device = os.getenv("EMBEDDINGS_DEVICE") or None  # None = auto-détection
    batch_size = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32"))

    try:
        logger.info("Démarrage du pipeline de création du vector store...")

        # Exécution du pipeline complet
        vector_store = create_vector_store_pipeline(
            save_path=save_path,
            limit=limit,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            verbose=True,
        )

        # Test de recherche (optionnel)
        test_query = os.getenv("TEST_QUERY")
        if test_query:
            logger.info(f"\n{'='*70}")
            logger.info("🔍 TEST DE RECHERCHE SÉMANTIQUE")
            logger.info(f"{'='*70}")
            logger.info(f"Requête: '{test_query}'")

            results = search_similar_documents(
                vector_store, test_query, k=5, verbose=True
            )

            logger.info(f"\n{'='*70}")
            logger.info("📊 RÉSULTATS DÉTAILLÉS")
            logger.info("=" * 70)

            for i, (doc, score) in enumerate(results, 1):
                logger.info(f"\n--- Résultat {i} (Score: {score:.4f}) ---")
                logger.info(f"Titre: {doc.metadata.get('title', 'N/A')}")
                logger.info(f"Lieu: {doc.metadata.get('locationName', 'N/A')}")
                logger.info(f"Date: {doc.metadata.get('dateRange', 'N/A')}")
                logger.info(f"Région: {doc.metadata.get('region', 'N/A')}")
                logger.info(f"\nExtrait: {doc.page_content[:300]}...")
                logger.info("-" * 70)

        logger.info("\n✓ Programme terminé avec succès")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
