"""
Pipeline complet pour la création de la base vectorielle depuis MongoDB.

Ce module orchestre l'ensemble du processus :
1. Connexion à MongoDB
2. Chargement et chunking des documents
   2.5. Suppression de l'index FAISS existant (après validation des données)
3. Génération des embeddings et création de l'index FAISS
4. Sauvegarde et test de recherche
"""

from typing import Optional, Dict, Any
import os
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from pymongo import MongoClient

from embeddings import get_embeddings_model
from vectors import (
    create_vector_store,
    save_vector_store,
    search_similar_documents,
    delete_vector_store,
)
from chunks.chunks_document import get_mongodb_connection, process_events_to_chunks

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_default_updated_date() -> str:
    """
    Calcule la date par défaut pour updatedAt.gte (aujourd'hui - 1 an).

    Returns:
        str: Date au format ISO 8601 (ex: "2024-11-03T00:00:00.000Z")
    """
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    return one_year_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def calculate_months_back(date_str: str) -> int:
    """
    Calcule le nombre de mois entre une date donnée et aujourd'hui.

    Args:
        date_str: Date au format ISO 8601 (ex: "2024-01-01T00:00:00.000Z")

    Returns:
        int: Nombre de mois approximatif (basé sur 30 jours par mois)
    """
    try:
        # Parser la date ISO 8601
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'

        # Conversion en datetime
        target_date = datetime.fromisoformat(date_str)
        now = datetime.now(timezone.utc)

        # Calculer la différence en jours
        days_diff = (now - target_date).days

        # Convertir en mois (approximation: 30 jours par mois)
        months = round(days_diff / 30)

        return max(0, months)  # Ne pas retourner de valeur négative
    except Exception as e:
        logger.warning(f"Erreur lors du calcul des mois: {e}")
        return 12  # Valeur par défaut: 12 mois


def get_last_execution_date(verbose: bool = False) -> str:
    """
    Récupère la date de la dernière exécution du pipeline depuis MongoDB.

    Args:
        verbose: Si True, affiche des informations

    Returns:
        str: Date de la dernière exécution au format ISO 8601, ou None si aucune exécution
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    client = None
    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        last_update_collection = db["last_update"]

        # Récupérer la dernière exécution
        last_execution = last_update_collection.find_one(
            {},
            sort=[("pipeline_run_date", -1)]
        )

        if last_execution and "pipeline_run_date" in last_execution:
            run_date = last_execution["pipeline_run_date"]
            # Convertir en format ISO 8601
            if isinstance(run_date, datetime):
                iso_date = run_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            else:
                iso_date = str(run_date)

            if verbose:
                logger.info(f"Dernière exécution trouvée: {iso_date}")
            return iso_date
        else:
            if verbose:
                logger.info("Aucune exécution précédente trouvée")
            return None

    except Exception as e:
        logger.warning(f"Erreur lors de la récupération de la dernière exécution: {e}")
        return None
    finally:
        if client:
            client.close()


def save_last_update_metadata(
    updated_at_gte: str,
    months_back: int,
    total_chunks: int,
    total_events: int,
    verbose: bool = False
) -> None:
    """
    Sauvegarde les métadonnées de la dernière mise à jour dans MongoDB.

    Args:
        updated_at_gte: Date de sélection utilisée pour filtrer les agendas
        months_back: Nombre de mois en arrière recherchés
        total_chunks: Nombre total de chunks créés
        total_events: Nombre total d'événements traités
        verbose: Si True, affiche des informations de progression
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    client = None
    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        last_update_collection = db["last_update"]

        # Préparer le document de métadonnées
        metadata = {
            "pipeline_run_date": datetime.now(timezone.utc),
            "agendas_updated_at_gte": updated_at_gte,
            "months_back": months_back,
            "total_events_processed": total_events,
            "total_chunks_created": total_chunks,
            "region": os.getenv("OA_REGION", "N/A"),
            "embeddings_model": os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        }

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("💾 SAUVEGARDE DES MÉTADONNÉES")
            logger.info("=" * 70)
            logger.info(f"Date de sélection: {updated_at_gte}")
            logger.info(f"Mois recherchés: {months_back}")
            logger.info(f"Événements traités: {total_events}")
            logger.info(f"Chunks créés: {total_chunks}")

        # Insérer le document (on garde l'historique)
        last_update_collection.insert_one(metadata)

        if verbose:
            logger.info("✅ Métadonnées sauvegardées dans la collection 'last_update'")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde des métadonnées: {e}", exc_info=True)
    finally:
        if client:
            client.close()


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

        # 2.5. Suppression de l'index FAISS existant (après vérification des données)
        if save_path:
            if verbose:
                logger.info("\n[2.5/4] Suppression de l'index FAISS existant...")
            delete_vector_store(save_path, verbose=verbose)

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

        return vector_store, len(chunks)

    finally:
        client.close()
        if verbose:
            logger.info("Connexion MongoDB fermée")


def main():
    """
    Fonction principale pour exécuter le pipeline complet.
    Configuration via variables d'environnement et arguments de ligne de commande.

    Arguments:
        mode: 'update' ou 'recreate' (défaut: 'recreate')
            - update: Mode incrémental, traite uniquement les nouveaux événements depuis la dernière exécution
            - recreate: Mode complet, recrée tout l'index depuis le début
    """
    import sys

    # Chargement des variables d'environnement
    load_dotenv()

    # Déterminer le mode (update ou recreate)
    mode = sys.argv[1] if len(sys.argv) > 1 else "recreate"

    if mode not in ["update", "recreate"]:
        logger.error(f"Mode invalide: {mode}. Utilisez 'update' ou 'recreate'")
        sys.exit(1)

    # Configuration
    save_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    limit = os.getenv("EMBEDDINGS_LIMIT")
    limit = int(limit) if limit else None

    model_id = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
    device = os.getenv("EMBEDDINGS_DEVICE") or None  # None = auto-détection
    batch_size = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32"))

    try:
        logger.info("=" * 70)
        logger.info(f"MODE: {mode.upper()}")
        logger.info("=" * 70)

        # Déterminer la date de filtrage selon le mode
        if mode == "update":
            # Mode incrémental: utiliser la date de la dernière exécution
            logger.info("Mode UPDATE: Recherche de la dernière exécution...")
            last_execution_date = get_last_execution_date(verbose=True)

            if last_execution_date:
                # Utiliser cette date comme date minimale de mise à jour
                os.environ["OA_AGENDAS_UPDATED_AT_GTE"] = last_execution_date
                logger.info(f"✓ Date de mise à jour minimale: {last_execution_date}")
                logger.info("  Seuls les événements nouveaux/modifiés seront traités")
            else:
                logger.warning("⚠️  Aucune exécution précédente trouvée")
                logger.info("   Passage en mode RECREATE (traitement complet)")
                mode = "recreate"

        if mode == "recreate":
            # Mode complet: utiliser la date par défaut ou celle du .env
            logger.info("Mode RECREATE: Reconstruction complète de l'index")
            updated_at_gte = os.getenv("OA_AGENDAS_UPDATED_AT_GTE")
            if updated_at_gte:
                logger.info(f"✓ Date de mise à jour minimale: {updated_at_gte} (depuis .env)")
            else:
                default_date = get_default_updated_date()
                logger.info(f"✓ Date de mise à jour minimale: {default_date} (par défaut: 1 an)")

        logger.info("=" * 70)
        logger.info("Démarrage du pipeline de création du vector store...")

        # Exécution du pipeline complet
        vector_store, total_chunks = create_vector_store_pipeline(
            save_path=save_path,
            limit=limit,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            verbose=True,
        )

        # Sauvegarde des métadonnées de mise à jour
        # Utiliser la valeur de .env ou calculer la date par défaut (1 an en arrière)
        updated_at_gte = os.getenv("OA_AGENDAS_UPDATED_AT_GTE")
        if not updated_at_gte:
            updated_at_gte = get_default_updated_date()
            logger.info(f"Variable OA_AGENDAS_UPDATED_AT_GTE non définie, utilisation de la date par défaut: {updated_at_gte}")

        months_back = calculate_months_back(updated_at_gte)

        # Compter le nombre d'événements dans MongoDB
        client_meta = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        db_meta = client_meta[os.getenv("MONGODB_DB_NAME", "OA")]
        events_collection_meta = db_meta[os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")]
        total_events = events_collection_meta.count_documents({})
        client_meta.close()

        save_last_update_metadata(
            updated_at_gte=updated_at_gte,
            months_back=months_back,
            total_chunks=total_chunks,
            total_events=total_events,
            verbose=True
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
