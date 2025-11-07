"""
Script pour supprimer les collections backup dans MongoDB.

Ce script identifie et supprime les collections de backup créées par le pipeline:
- Collections avec suffix _backup_YYYYMMDD_HHMMSS (full backup)
- Collections avec suffix _update_YYYYMMDD_HHMMSS (update backup)

Usage:
    python src/utils/clean_backups.py              # Mode interactif (demande confirmation)
    python src/utils/clean_backups.py --dry-run    # Affiche sans supprimer
    python src/utils/clean_backups.py --force      # Supprime sans confirmation
"""

import os
import re
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_mongodb_connection():
    """
    Établit une connexion à MongoDB.

    Returns:
        tuple: (client, database) ou (None, None) en cas d'erreur
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")

    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        # Test de connexion
        client.server_info()
        return client, db
    except Exception as e:
        logger.error(f"❌ Erreur de connexion à MongoDB: {e}")
        return None, None


def is_backup_collection(collection_name: str) -> bool:
    """
    Vérifie si une collection est une collection backup.

    Args:
        collection_name: Nom de la collection

    Returns:
        bool: True si c'est une collection backup
    """
    # Pattern pour _backup_YYYYMMDD_HHMMSS
    backup_pattern = r".*_backup_\d{8}_\d{6}$"
    # Pattern pour _update_YYYYMMDD_HHMMSS
    update_pattern = r".*_update_\d{8}_\d{6}$"

    return (
        re.match(backup_pattern, collection_name) is not None
        or re.match(update_pattern, collection_name) is not None
    )


def extract_backup_info(collection_name: str) -> Dict[str, str]:
    """
    Extrait les informations d'une collection backup.

    Args:
        collection_name: Nom de la collection

    Returns:
        dict: Informations (type, date, original_name)
    """
    # Pattern avec capture de groupes
    pattern = r"(.+)_(backup|update)_(\d{8})_(\d{6})$"
    match = re.match(pattern, collection_name)

    if match:
        original_name = match.group(1)
        backup_type = match.group(2)
        date_str = match.group(3)
        time_str = match.group(4)

        # Convertir en datetime
        try:
            dt = datetime.strptime(
                f"{date_str}_{time_str}",
                "%Y%m%d_%H%M%S"
            )
            date_formatted = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            date_formatted = f"{date_str}_{time_str}"

        return {
            "original_name": original_name,
            "type": backup_type,
            "date": date_formatted,
            "timestamp": f"{date_str}_{time_str}"
        }

    return {
        "original_name": "unknown",
        "type": "unknown",
        "date": "unknown",
        "timestamp": "unknown"
    }


def list_backup_collections(db) -> List[Dict[str, any]]:
    """
    Liste toutes les collections backup dans la base de données.

    Args:
        db: Base de données MongoDB

    Returns:
        list: Liste de dictionnaires avec infos des backups
    """
    all_collections = db.list_collection_names()
    backup_collections = []

    for collection_name in all_collections:
        if is_backup_collection(collection_name):
            info = extract_backup_info(collection_name)
            collection = db[collection_name]
            doc_count = collection.count_documents({})

            backup_collections.append({
                "name": collection_name,
                "original_name": info["original_name"],
                "type": info["type"],
                "date": info["date"],
                "timestamp": info["timestamp"],
                "count": doc_count
            })

    # Trier par timestamp (plus récent en premier)
    backup_collections.sort(key=lambda x: x["timestamp"], reverse=True)

    return backup_collections


def display_backup_collections(backups: List[Dict[str, any]]):
    """
    Affiche la liste des collections backup de manière formatée.

    Args:
        backups: Liste des collections backup
    """
    if not backups:
        logger.info("✅ Aucune collection backup trouvée")
        return

    logger.info("\n" + "=" * 80)
    logger.info("📦 COLLECTIONS BACKUP DÉTECTÉES")
    logger.info("=" * 80)
    logger.info(f"Total: {len(backups)} collection(s) backup\n")

    # Grouper par type
    backup_by_type = {"backup": [], "update": []}
    for backup in backups:
        backup_by_type[backup["type"]].append(backup)

    # Afficher par type
    for backup_type in ["backup", "update"]:
        collections = backup_by_type[backup_type]
        if not collections:
            continue

        type_label = "FULL BACKUP" if backup_type == "backup" else "UPDATE BACKUP"
        logger.info(f"📁 {type_label} ({len(collections)} collection(s))")
        logger.info("-" * 80)

        for i, backup in enumerate(collections, 1):
            logger.info(
                f"{i:2d}. {backup['name']}"
            )
            logger.info(
                f"    Collection source: {backup['original_name']}"
            )
            logger.info(
                f"    Date de création: {backup['date']}"
            )
            logger.info(
                f"    Documents: {backup['count']:,}"
            )
            logger.info("")

    logger.info("=" * 80)


def delete_backup_collections(
    db,
    backups: List[Dict[str, any]],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Supprime les collections backup.

    Args:
        db: Base de données MongoDB
        backups: Liste des collections backup à supprimer
        dry_run: Si True, n'effectue pas la suppression réelle

    Returns:
        dict: Statistiques de suppression
    """
    stats = {
        "total": len(backups),
        "deleted": 0,
        "failed": 0,
        "total_docs": 0
    }

    if dry_run:
        logger.info("\n🔍 MODE DRY-RUN - Aucune suppression ne sera effectuée")
        stats["total_docs"] = sum(b["count"] for b in backups)
        return stats

    logger.info("\n🗑️  SUPPRESSION DES COLLECTIONS BACKUP")
    logger.info("=" * 80)

    for i, backup in enumerate(backups, 1):
        collection_name = backup["name"]
        doc_count = backup["count"]

        try:
            logger.info(
                f"[{i}/{stats['total']}] Suppression de '{collection_name}' "
                f"({doc_count:,} documents)..."
            )
            db.drop_collection(collection_name)
            stats["deleted"] += 1
            stats["total_docs"] += doc_count
            logger.info("    ✅ Collection supprimée")

        except Exception as e:
            stats["failed"] += 1
            logger.error(f"    ❌ Échec: {e}")

    return stats


def confirm_deletion(backups: List[Dict[str, any]]) -> bool:
    """
    Demande confirmation à l'utilisateur avant suppression.

    Args:
        backups: Liste des collections backup à supprimer

    Returns:
        bool: True si l'utilisateur confirme
    """
    total_docs = sum(b["count"] for b in backups)

    logger.info("\n⚠️  CONFIRMATION DE SUPPRESSION")
    logger.info("=" * 80)
    logger.info(f"Nombre de collections à supprimer: {len(backups)}")
    logger.info(f"Nombre total de documents: {total_docs:,}")
    logger.info("=" * 80)
    logger.info("Cette action est IRRÉVERSIBLE !")
    logger.info("")

    try:
        response = input("Confirmer la suppression ? (oui/non): ").strip().lower()
        return response in ["oui", "o", "yes", "y"]
    except KeyboardInterrupt:
        logger.info("\n\n❌ Opération annulée par l'utilisateur")
        return False


def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(
        description="Supprime les collections backup dans MongoDB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les collections sans les supprimer"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Supprime sans demander confirmation"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🧹 NETTOYAGE DES COLLECTIONS BACKUP MONGODB")
    logger.info("=" * 80)

    # Connexion MongoDB
    client, db = get_mongodb_connection()
    if not client:
        sys.exit(1)

    try:
        # Lister les backups
        backups = list_backup_collections(db)
        display_backup_collections(backups)

        if not backups:
            return

        # Mode dry-run
        if args.dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("🔍 MODE DRY-RUN ACTIVÉ")
            logger.info("=" * 80)
            logger.info("Aucune suppression ne sera effectuée")
            logger.info(
                f"Collections qui seraient supprimées: {len(backups)}"
            )
            logger.info(
                f"Documents qui seraient supprimés: "
                f"{sum(b['count'] for b in backups):,}"
            )
            logger.info("=" * 80)
            return

        # Demander confirmation (sauf si --force)
        if not args.force:
            if not confirm_deletion(backups):
                logger.info("❌ Opération annulée")
                return

        # Supprimer les backups
        stats = delete_backup_collections(db, backups, dry_run=False)

        # Afficher résumé
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DE LA SUPPRESSION")
        logger.info("=" * 80)
        logger.info(f"Collections à traiter: {stats['total']}")
        logger.info(f"Collections supprimées: {stats['deleted']}")
        logger.info(f"Échecs: {stats['failed']}")
        logger.info(f"Documents supprimés: {stats['total_docs']:,}")
        logger.info("=" * 80)

        if stats['deleted'] > 0:
            logger.info("✅ Nettoyage terminé avec succès")
        else:
            logger.warning("⚠️  Aucune collection n'a été supprimée")

    except KeyboardInterrupt:
        logger.info("\n\n❌ Opération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if client:
            client.close()
            logger.info("🔌 Connexion MongoDB fermée")


if __name__ == "__main__":
    main()
