"""
Script pour afficher les métadonnées de la dernière exécution du pipeline.

Ce script lit la collection 'last_update' de MongoDB et affiche les paramètres
de la dernière exécution du pipeline de génération d'embeddings.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_date(iso_date) -> str:
    """
    Formate une date ISO en format lisible.

    Args:
        iso_date: Date au format ISO ou datetime

    Returns:
        str: Date formatée (ex: "03/11/2024 14:30:25")
    """
    if isinstance(iso_date, datetime):
        return iso_date.strftime("%d/%m/%Y %H:%M:%S")
    elif isinstance(iso_date, str):
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            return iso_date
    return str(iso_date)


def show_last_update(verbose: bool = True) -> dict:
    """
    Affiche les métadonnées de la dernière exécution du pipeline.

    Args:
        verbose: Si True, affiche les informations formatées

    Returns:
        dict: Métadonnées de la dernière exécution ou None si aucune
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    client = None
    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        last_update_collection = db["last_update"]

        # Compter le nombre d'exécutions
        total_executions = last_update_collection.count_documents({})

        if total_executions == 0:
            if verbose:
                logger.info("=" * 70)
                logger.info("ℹ️  Aucune exécution du pipeline enregistrée")
                logger.info("=" * 70)
            return None

        # Récupérer la dernière exécution (tri par date décroissante)
        last_execution = last_update_collection.find_one(
            {},
            sort=[("pipeline_run_date", -1)]
        )

        if not last_execution:
            if verbose:
                logger.info("=" * 70)
                logger.info("ℹ️  Aucune exécution du pipeline enregistrée")
                logger.info("=" * 70)
            return None

        if verbose:
            logger.info("=" * 70)
            logger.info("📊 DERNIÈRE EXÉCUTION DU PIPELINE")
            logger.info("=" * 70)
            logger.info("")
            logger.info("🗓️  Date d'exécution:")
            logger.info(
                f"   {format_date(last_execution.get('pipeline_run_date'))}"
            )
            logger.info("")
            logger.info("📅 Paramètres de sélection:")
            logger.info(
                f"   Date de début: "
                f"{last_execution.get('agendas_updated_at_gte', 'N/A')}"
            )
            logger.info(
                f"   Mois recherchés: "
                f"{last_execution.get('months_back', 'N/A')}"
            )
            logger.info(
                f"   Région: {last_execution.get('region', 'N/A')}"
            )
            logger.info("")
            logger.info("📊 Données traitées:")
            logger.info(
                f"   Événements: "
                f"{last_execution.get('total_events_processed', 0):,}"
            )
            logger.info(
                f"   Chunks créés: "
                f"{last_execution.get('total_chunks_created', 0):,}"
            )
            logger.info("")
            logger.info("🤖 Configuration du modèle:")
            logger.info(
                f"   Modèle: "
                f"{last_execution.get('embeddings_model', 'N/A')}"
            )
            logger.info(
                f"   Chunk size: {last_execution.get('chunk_size', 'N/A')}"
            )
            logger.info(
                f"   Chunk overlap: "
                f"{last_execution.get('chunk_overlap', 'N/A')}"
            )
            logger.info("")
            logger.info("📈 Historique:")
            logger.info(f"   Total d'exécutions: {total_executions}")
            logger.info("=" * 70)

        return last_execution

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture des métadonnées: {e}", exc_info=True)
        return None
    finally:
        if client:
            client.close()


def show_execution_history(limit: int = 5, verbose: bool = True) -> list:
    """
    Affiche l'historique des dernières exécutions du pipeline.

    Args:
        limit: Nombre d'exécutions à afficher
        verbose: Si True, affiche les informations formatées

    Returns:
        list: Liste des métadonnées des dernières exécutions
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    client = None
    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        last_update_collection = db["last_update"]

        # Récupérer les dernières exécutions
        executions = list(
            last_update_collection.find(
                {},
                sort=[("pipeline_run_date", -1)],
                limit=limit
            )
        )

        if not executions:
            if verbose:
                logger.info("=" * 70)
                logger.info("ℹ️  Aucune exécution du pipeline enregistrée")
                logger.info("=" * 70)
            return []

        if verbose:
            logger.info("=" * 70)
            logger.info(f"📜 HISTORIQUE DES DERNIÈRES EXÉCUTIONS (Top {len(executions)})")
            logger.info("=" * 70)
            logger.info("")

            for i, execution in enumerate(executions, 1):
                logger.info(f"{'─' * 70}")
                logger.info(f"Exécution #{i}")
                logger.info(f"{'─' * 70}")
                logger.info(f"Date: {format_date(execution.get('pipeline_run_date'))}")
                logger.info(
                    f"Événements: {execution.get('total_events_processed', 0):,} | "
                    f"Chunks: {execution.get('total_chunks_created', 0):,}"
                )
                logger.info(
                    f"Période: {execution.get('months_back', 'N/A')} mois | "
                    f"Région: {execution.get('region', 'N/A')}"
                )
                logger.info("")

            logger.info("=" * 70)

        return executions

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture de l'historique: {e}", exc_info=True)
        return []
    finally:
        if client:
            client.close()


def main():
    """
    Point d'entrée principal pour afficher les métadonnées.
    """
    import sys

    # Vérifier si on demande l'historique
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        show_execution_history(limit=limit, verbose=True)
    else:
        show_last_update(verbose=True)


if __name__ == "__main__":
    main()
