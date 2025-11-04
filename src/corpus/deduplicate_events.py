"""
Script de dédoublonnement de la collection MongoDB events.

Ce script supprime les événements en double basés sur la clé 'uid'.
En cas de doublons, il conserve le document le plus récent (basé sur 'updatedAt').
"""

import os
import logging
from typing import Dict, Any
from collections import defaultdict
from pymongo import MongoClient
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
    Établit la connexion à MongoDB et retourne la collection events.

    Returns:
        tuple: (MongoClient, Collection) - Client et collection events
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "OA")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")

    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    logger.info(f"Connexion à MongoDB: {mongodb_uri}")
    logger.info(f"Base de données: {db_name}")
    logger.info(f"Collection: {collection_name}")

    return client, collection


def find_duplicates(collection) -> Dict[str, list]:
    """
    Trouve tous les événements en double basés sur le champ 'uid'.

    Args:
        collection: Collection MongoDB

    Returns:
        dict: Dictionnaire {uid: [list of document _ids]} pour les doublons
    """
    logger.info("Recherche des doublons...")

    # Utiliser une agrégation pour trouver les uid en double
    pipeline = [
        {
            "$group": {
                "_id": "$uid",
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"},
                "updatedAts": {"$push": "$updatedAt"}
            }
        },
        {
            "$match": {
                "count": {"$gt": 1}
            }
        }
    ]

    duplicates = {}
    for result in collection.aggregate(pipeline):
        uid = result["_id"]
        duplicates[uid] = {
            "ids": result["ids"],
            "updatedAts": result["updatedAts"],
            "count": result["count"]
        }

    return duplicates


def deduplicate_events(collection, dry_run: bool = False) -> Dict[str, int]:
    """
    Supprime les événements en double de la collection.

    Pour chaque uid en double, conserve le document avec la date
    'updatedAt' la plus récente et supprime les autres.

    Args:
        collection: Collection MongoDB
        dry_run: Si True, simule sans supprimer (défaut: False)

    Returns:
        dict: Statistiques de dédoublonnement
    """
    stats = {
        "total_events": 0,
        "duplicate_uids": 0,
        "duplicate_documents": 0,
        "documents_to_delete": 0,
        "documents_deleted": 0,
    }

    # Compter le nombre total d'événements
    stats["total_events"] = collection.count_documents({})
    logger.info(f"Nombre total d'événements: {stats['total_events']}")

    # Trouver les doublons
    duplicates = find_duplicates(collection)
    stats["duplicate_uids"] = len(duplicates)

    if stats["duplicate_uids"] == 0:
        logger.info("✅ Aucun doublon trouvé dans la collection")
        return stats

    logger.info(f"⚠️  {stats['duplicate_uids']} uid en double trouvés")

    # Pour chaque uid en double, identifier les documents à supprimer
    ids_to_delete = []

    for uid, data in duplicates.items():
        ids = data["ids"]
        updated_ats = data["updatedAts"]
        count = data["count"]

        stats["duplicate_documents"] += count

        # Créer une liste de tuples (id, updatedAt) pour trier
        id_date_pairs = list(zip(ids, updated_ats))

        # Trier par updatedAt (le plus récent en premier)
        # Gérer les cas où updatedAt peut être None
        id_date_pairs.sort(key=lambda x: x[1] if x[1] else "", reverse=True)

        # Garder le premier (le plus récent), supprimer les autres
        for doc_id, updated_at in id_date_pairs[1:]:
            ids_to_delete.append(doc_id)
            logger.debug(f"  - uid={uid}, _id={doc_id}, updatedAt={updated_at} -> À SUPPRIMER")

        # Log du document conservé
        kept_id, kept_date = id_date_pairs[0]
        logger.debug(f"  ✓ uid={uid}, _id={kept_id}, updatedAt={kept_date} -> CONSERVÉ")

    stats["documents_to_delete"] = len(ids_to_delete)

    logger.info(f"📊 {stats['duplicate_documents']} documents en double au total")
    logger.info(f"🗑️  {stats['documents_to_delete']} documents à supprimer")
    logger.info(f"✅ {stats['duplicate_uids']} documents seront conservés (les plus récents)")

    if dry_run:
        logger.info("🔍 MODE DRY-RUN: Aucune suppression effectuée")
        return stats

    # Supprimer les doublons
    if ids_to_delete:
        logger.info("Suppression des doublons en cours...")
        result = collection.delete_many({"_id": {"$in": ids_to_delete}})
        stats["documents_deleted"] = result.deleted_count
        logger.info(f"✅ {stats['documents_deleted']} documents supprimés")

        # Vérifier le nouveau total
        new_total = collection.count_documents({})
        logger.info(f"📊 Nombre d'événements après dédoublonnement: {new_total}")
        logger.info(f"📉 Réduction: {stats['total_events'] - new_total} documents")

    return stats


def main():
    """
    Point d'entrée principal du script de dédoublonnement.
    """
    logger.info("=" * 70)
    logger.info("DÉDOUBLONNEMENT DE LA COLLECTION EVENTS")
    logger.info("=" * 70)

    client = None
    try:
        # Connexion à MongoDB
        client, collection = get_mongodb_connection()

        # Exécuter le dédoublonnement
        stats = deduplicate_events(collection, dry_run=False)

        # Afficher le résumé
        logger.info("=" * 70)
        logger.info("RÉSUMÉ DU DÉDOUBLONNEMENT")
        logger.info("=" * 70)
        logger.info(f"Events avant dédoublonnement: {stats['total_events']}")
        logger.info(f"UID en double: {stats['duplicate_uids']}")
        logger.info(f"Documents en double: {stats['duplicate_documents']}")
        logger.info(f"Documents supprimés: {stats['documents_deleted']}")
        logger.info(f"Events après dédoublonnement: {stats['total_events'] - stats['documents_deleted']}")
        logger.info("=" * 70)

        if stats['documents_deleted'] > 0:
            logger.info("✅ Dédoublonnement terminé avec succès")
        else:
            logger.info("✅ Aucun doublon à supprimer")

    except Exception as e:
        logger.error(f"❌ Erreur lors du dédoublonnement: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()
            logger.info("Connexion MongoDB fermée")


if __name__ == "__main__":
    main()
