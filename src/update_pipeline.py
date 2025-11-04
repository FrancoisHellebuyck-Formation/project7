"""
Pipeline de mise à jour incrémentale complet.

Ce module orchestre la mise à jour incrémentale complète :
1. Récupération de la date de dernière exécution
2. Récupération des agendas mis à jour depuis cette date
3. Récupération des événements pour ces agendas
4. Dédoublonnement des événements
5. Chunking des documents
6. Génération des embeddings et mise à jour FAISS

Mode UPDATE : Traite uniquement les nouveaux/modifiés depuis la dernière exécution
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_last_execution_date() -> str:
    """
    Récupère la date de la dernière exécution du pipeline depuis MongoDB.

    Returns:
        str: Date de la dernière exécution au format ISO 8601, ou None si aucune exécution
    """
    from pymongo import MongoClient

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
            {}, sort=[("pipeline_run_date", -1)]
        )

        if last_execution and "pipeline_run_date" in last_execution:
            run_date = last_execution["pipeline_run_date"]
            # Convertir en format ISO 8601
            if isinstance(run_date, datetime):
                iso_date = run_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            else:
                iso_date = str(run_date)

            logger.info(f"✓ Dernière exécution trouvée: {iso_date}")
            return iso_date
        else:
            logger.warning("⚠️  Aucune exécution précédente trouvée")
            return None

    except Exception as e:
        logger.error(
            f"❌ Erreur lors de la récupération de la dernière exécution: {e}",
            exc_info=True,
        )
        return None
    finally:
        if client:
            client.close()


def run_command(command: list, description: str) -> bool:
    """
    Exécute une commande shell et affiche le résultat.

    Args:
        command: Liste des arguments de la commande
        description: Description de l'étape

    Returns:
        bool: True si succès, False si échec
    """
    logger.info("=" * 70)
    logger.info(f"📋 {description}")
    logger.info("=" * 70)

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
            env=os.environ.copy()
        )
        logger.info(f"✅ {description} - TERMINÉ")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - ÉCHEC")
        logger.error(f"Code de retour: {e.returncode}")
        return False


def main():
    """
    Fonction principale pour exécuter le pipeline de mise à jour incrémentale.
    """
    load_dotenv()

    logger.info("\n" + "=" * 70)
    logger.info("🔄 PIPELINE DE MISE À JOUR INCRÉMENTALE")
    logger.info("=" * 70)
    logger.info("Ce pipeline va :")
    logger.info("  1. Récupérer la date de dernière exécution")
    logger.info("  2. Sauvegarder et vider les collections agendas/events")
    logger.info("  3. Récupérer les agendas mis à jour depuis cette date")
    logger.info("  4. Récupérer les événements pour ces agendas")
    logger.info("  5. Dédoublonner les événements")
    logger.info("  6. Chunker les documents")
    logger.info("  7. Générer les embeddings et mettre à jour FAISS")
    logger.info("=" * 70 + "\n")

    # Étape 1 : Récupérer la date de dernière exécution
    logger.info("[1/7] Récupération de la date de dernière exécution...")
    last_execution_date = get_last_execution_date()

    if not last_execution_date:
        logger.error("❌ Aucune exécution précédente trouvée.")
        logger.error(
            "   Veuillez d'abord exécuter 'make run-all' pour créer l'index initial."
        )
        sys.exit(1)

    # Définir les variables d'environnement
    os.environ["OA_AGENDAS_UPDATED_AT_GTE"] = last_execution_date
    os.environ["OA_EVENTS_DATE_FILTER"] = last_execution_date
    logger.info(
        f"✓ Date de mise à jour minimale définie: "
        f"{last_execution_date}"
    )
    logger.info(
        "  → Les agendas modifiés depuis cette date seront "
        "récupérés"
    )
    logger.info(
        "  → Les événements créés ou mis à jour depuis cette date "
        "seront inclus"
    )

    # Étape 2 : Backup et vidage des collections agendas/events
    logger.info("\n[2/7] Sauvegarde et vidage des collections...")
    try:
        from corpus.cleanup_mongodb import backup_and_clear_for_update

        backup_and_clear_for_update(verbose=True)
        logger.info("✅ Collections sauvegardées et vidées")
    except Exception as e:
        logger.error(f"❌ Échec du backup: {e}")
        sys.exit(1)

    # Étape 3 : Récupération des agendas
    if not run_command(
        ["uv", "run", "python", "src/corpus/get_corpus_agendas.py"],
        "[3/7] Récupération des agendas mis à jour",
    ):
        logger.error("❌ Échec de la récupération des agendas")
        sys.exit(1)

    # Étape 4 : Récupération des événements
    if not run_command(
        ["uv", "run", "python", "src/corpus/get_corpus_events.py"],
        "[4/7] Récupération des événements",
    ):
        logger.error("❌ Échec de la récupération des événements")
        sys.exit(1)

    # Étape 5 : Dédoublonnement
    if not run_command(
        ["uv", "run", "python", "src/corpus/deduplicate_events.py"],
        "[5/7] Dédoublonnement des événements",
    ):
        logger.error("❌ Échec du dédoublonnement")
        sys.exit(1)

    # Étape 6 : Chunking (pas de script séparé, intégré dans pipeline.py)
    logger.info("=" * 70)
    logger.info("[6/7] Chunking des documents (intégré dans étape 7)")
    logger.info("=" * 70)

    # Étape 7 : Génération des embeddings (mode update)
    # Le pipeline.py en mode update va :
    # - Charger les documents depuis MongoDB
    # - Créer les chunks
    # - Supprimer l'ancien index FAISS
    # - Créer le nouvel index avec tous les documents
    if not run_command(
        ["uv", "run", "python", "src/pipeline.py", "update"],
        "[7/7] Génération des embeddings et mise à jour FAISS",
    ):
        logger.error("❌ Échec de la génération des embeddings")
        sys.exit(1)

    # Succès
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE DE MISE À JOUR TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 70)
    logger.info(
        f"Date de mise à jour utilisée: {last_execution_date}"
    )
    logger.info("Consultez les logs ci-dessus pour les détails.")
    logger.info("\n💡 Utilisez 'make show-last-update' pour voir les statistiques")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
