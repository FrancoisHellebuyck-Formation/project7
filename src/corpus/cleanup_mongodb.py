"""
Script de nettoyage de la base MongoDB.

Ce script archive les collections existantes en les renommant avec la date du jour,
permettant de démarrer une nouvelle génération de corpus avec des collections propres.

Les collections sont renommées selon le format:
- agendas → agendas_backup_YYYYMMDD_HHMMSS
- events → events_backup_YYYYMMDD_HHMMSS
"""

import os
import logging
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_mongodb_connection():
    """
    Établit la connexion à MongoDB et retourne le client et la base de données.

    Returns:
        tuple: (MongoClient, Database) - Client et base de données
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    client = MongoClient(mongodb_uri)
    db = client[db_name]

    logger.info(f"Connexion à MongoDB: {mongodb_uri}")
    logger.info(f"Base de données: {db_name}")

    return client, db


def get_backup_timestamp() -> str:
    """
    Génère un timestamp pour le suffixe de backup.

    Returns:
        str: Timestamp au format YYYYMMDD_HHMMSS (ex: "20241103_143025")
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S")


def collection_exists(db, collection_name: str) -> bool:
    """
    Vérifie si une collection existe dans la base de données.

    Args:
        db: Base de données MongoDB
        collection_name: Nom de la collection

    Returns:
        bool: True si la collection existe, False sinon
    """
    return collection_name in db.list_collection_names()


def get_collection_stats(db, collection_name: str) -> dict:
    """
    Récupère les statistiques d'une collection.

    Args:
        db: Base de données MongoDB
        collection_name: Nom de la collection

    Returns:
        dict: Statistiques de la collection (count, size, etc.)
    """
    if not collection_exists(db, collection_name):
        return {"exists": False, "count": 0, "size": 0}

    collection = db[collection_name]
    count = collection.count_documents({})

    # Obtenir la taille via stats (peut échouer si la collection est vide)
    try:
        stats = db.command("collstats", collection_name)
        size = stats.get("size", 0)
    except OperationFailure:
        size = 0

    return {
        "exists": True,
        "count": count,
        "size": size,
    }


def rename_collection(db, old_name: str, new_name: str) -> bool:
    """
    Renomme une collection MongoDB.

    Args:
        db: Base de données MongoDB
        old_name: Nom actuel de la collection
        new_name: Nouveau nom de la collection

    Returns:
        bool: True si le renommage a réussi, False sinon
    """
    try:
        db[old_name].rename(new_name, dropTarget=True)
        logger.info(f"✅ Collection '{old_name}' renommée en '{new_name}'")
        return True
    except OperationFailure as e:
        logger.error(f"❌ Erreur lors du renommage de '{old_name}': {e}")
        return False


def cleanup_mongodb(dry_run: bool = False) -> dict:
    """
    Archive les collections existantes en les renommant avec la date du jour.

    Args:
        dry_run: Si True, simule sans renommer (défaut: False)

    Returns:
        dict: Statistiques du nettoyage
    """
    load_dotenv()

    agendas_collection_name = os.getenv("MONGODB_COLLECTION_NAME_AGENDAS", "agendas")
    events_collection_name = os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")

    stats = {
        "timestamp": get_backup_timestamp(),
        "collections_renamed": 0,
        "collections_not_found": 0,
        "agendas": {"renamed": False, "count": 0},
        "events": {"renamed": False, "count": 0},
    }

    client = None
    try:
        # Connexion à MongoDB
        client, db = get_mongodb_connection()

        # Générer le suffixe de backup
        backup_suffix = f"_backup_{stats['timestamp']}"

        # Liste des collections à traiter
        collections_to_rename = [
            (agendas_collection_name, f"{agendas_collection_name}{backup_suffix}"),
            (events_collection_name, f"{events_collection_name}{backup_suffix}"),
        ]

        logger.info("=" * 70)
        logger.info("NETTOYAGE DE LA BASE MONGODB")
        logger.info("=" * 70)

        # Afficher l'état des collections avant nettoyage
        for old_name, new_name in collections_to_rename:
            collection_stats = get_collection_stats(db, old_name)

            if collection_stats["exists"]:
                logger.info(f"📦 Collection '{old_name}':")
                logger.info(f"   - Documents: {collection_stats['count']:,}")
                logger.info(f"   - Taille: {collection_stats['size']:,} bytes")
                logger.info(f"   → Sera renommée en '{new_name}'")

                # Stocker les stats pour le résumé
                collection_key = old_name.replace(agendas_collection_name, "agendas").replace(events_collection_name, "events")
                if collection_key in stats:
                    stats[collection_key]["count"] = collection_stats["count"]
            else:
                logger.info(f"ℹ️  Collection '{old_name}' n'existe pas (sera créée)")
                stats["collections_not_found"] += 1

        logger.info("=" * 70)

        if dry_run:
            logger.info("🔍 MODE DRY-RUN: Aucun renommage effectué")
            return stats

        # Renommer les collections
        logger.info("Renommage des collections en cours...")
        logger.info("")

        for old_name, new_name in collections_to_rename:
            if collection_exists(db, old_name):
                success = rename_collection(db, old_name, new_name)

                if success:
                    stats["collections_renamed"] += 1
                    collection_key = old_name.replace(agendas_collection_name, "agendas").replace(events_collection_name, "events")
                    if collection_key in stats:
                        stats[collection_key]["renamed"] = True

        return stats

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()
            logger.info("Connexion MongoDB fermée")


def backup_and_clear_for_update(verbose: bool = True) -> dict:
    """
    Backup spécifique pour les mises à jour incrémentales.

    Archive les collections agendas et events, puis les vide.
    Ne touche PAS à la collection last_update (nécessaire pour connaître la dernière date).

    Args:
        verbose: Si True, affiche des informations détaillées

    Returns:
        dict: Statistiques du backup et nettoyage
    """
    load_dotenv()

    agendas_collection_name = os.getenv("MONGODB_COLLECTION_NAME_AGENDAS", "agendas")
    events_collection_name = os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")

    stats = {
        "timestamp": get_backup_timestamp(),
        "collections_backed_up": 0,
        "collections_cleared": 0,
        "agendas": {"backed_up": False, "cleared": False, "count": 0},
        "events": {"backed_up": False, "cleared": False, "count": 0},
    }

    client = None
    try:
        # Connexion à MongoDB
        client, db = get_mongodb_connection()

        # Générer le suffixe de backup
        backup_suffix = f"_update_{stats['timestamp']}"

        if verbose:
            logger.info("=" * 70)
            logger.info("BACKUP POUR MISE À JOUR INCRÉMENTALE")
            logger.info("=" * 70)
            logger.info("⚠️  Les collections agendas et events vont être archivées puis vidées")
            logger.info("✓  La collection last_update sera préservée (contient la date de dernière exécution)")
            logger.info("=" * 70)

        # Traiter les collections agendas et events
        collections_to_process = [
            (agendas_collection_name, "agendas"),
            (events_collection_name, "events"),
        ]

        for collection_name, stats_key in collections_to_process:
            if collection_exists(db, collection_name):
                collection = db[collection_name]
                count = collection.count_documents({})

                if count > 0:
                    backup_name = f"{collection_name}{backup_suffix}"

                    if verbose:
                        logger.info(f"\n📦 Collection '{collection_name}':")
                        logger.info(f"   - Documents: {count:,}")
                        logger.info(f"   → Backup en '{backup_name}'")

                    # Backup: renommer la collection
                    success = rename_collection(db, collection_name, backup_name)

                    if success:
                        stats["collections_backed_up"] += 1
                        stats[stats_key]["backed_up"] = True
                        stats[stats_key]["count"] = count

                        if verbose:
                            logger.info("   ✅ Backup créé")

                        # La collection originale n'existe plus (renommée)
                        # Elle sera recréée automatiquement lors de l'insertion
                        stats["collections_cleared"] += 1
                        stats[stats_key]["cleared"] = True

                        if verbose:
                            logger.info(
                                "   ✅ Collection vidée (sera recréée)"
                            )
                else:
                    if verbose:
                        logger.info(
                            f"\nℹ️  Collection '{collection_name}' vide - "
                            "aucun backup nécessaire"
                        )
            else:
                if verbose:
                    logger.info(
                        f"\nℹ️  Collection '{collection_name}' n'existe pas - "
                        "sera créée"
                    )

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("✅ BACKUP ET NETTOYAGE TERMINÉS")
            logger.info("=" * 70)
            logger.info(f"Collections sauvegardées: {stats['collections_backed_up']}")
            logger.info(f"Collections vidées: {stats['collections_cleared']}")
            logger.info("=" * 70)

        return stats

    except Exception as e:
        logger.error(f"❌ Erreur lors du backup: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()


def main():
    """
    Point d'entrée principal du script de nettoyage.
    """
    logger.info("=" * 70)
    logger.info("ARCHIVAGE DES COLLECTIONS MONGODB")
    logger.info("=" * 70)

    try:
        # Exécuter le nettoyage
        stats = cleanup_mongodb(dry_run=False)

        # Afficher le résumé
        logger.info("")
        logger.info("=" * 70)
        logger.info("RÉSUMÉ DU NETTOYAGE")
        logger.info("=" * 70)
        logger.info(f"Timestamp de backup: {stats['timestamp']}")
        logger.info(f"Collections renommées: {stats['collections_renamed']}")
        logger.info(f"Collections non trouvées: {stats['collections_not_found']}")
        logger.info("")

        if stats["agendas"]["renamed"]:
            logger.info(f"✅ Agendas archivés: {stats['agendas']['count']:,} documents")
        else:
            logger.info("ℹ️  Agendas: aucune collection à archiver")

        if stats["events"]["renamed"]:
            logger.info(f"✅ Events archivés: {stats['events']['count']:,} documents")
        else:
            logger.info("ℹ️  Events: aucune collection à archiver")

        logger.info("=" * 70)

        if stats["collections_renamed"] > 0:
            logger.info("✅ Nettoyage terminé avec succès")
            logger.info("💡 Les nouvelles collections seront créées lors de la prochaine exécution du corpus")
        else:
            logger.info("ℹ️  Aucune collection à archiver - base propre")

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
