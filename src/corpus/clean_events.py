"""
Script de nettoyage de la collection MongoDB events.

Ce script supprime les événements avec une longDescription trop courte
(inférieure à 100 caractères). Les événements avec peu de contenu sont
considérés comme insuffisants pour une recherche sémantique de qualité.
"""

import os
import logging
from typing import Dict
from pymongo import MongoClient
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Longueur minimale de la longDescription (en caractères)
MIN_DESCRIPTION_LENGTH = 100


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


def clean_events(collection, min_length: int = MIN_DESCRIPTION_LENGTH,
                 dry_run: bool = False) -> Dict[str, int]:
    """
    Supprime les événements avec une longDescription trop courte.

    Args:
        collection: Collection MongoDB
        min_length: Longueur minimale de longDescription (défaut: 100)
        dry_run: Si True, simule sans supprimer (défaut: False)

    Returns:
        dict: Statistiques de nettoyage
    """
    stats = {
        "total_events": 0,
        "events_with_short_description": 0,
        "events_without_description": 0,
        "events_to_delete": 0,
        "events_deleted": 0,
    }

    # Compter le nombre total d'événements
    stats["total_events"] = collection.count_documents({})
    logger.info(f"Nombre total d'événements: {stats['total_events']}")

    # Trouver les événements sans longDescription
    events_without_description = collection.count_documents({
        "$or": [
            {"longDescription": {"$exists": False}},
            {"longDescription": {"$eq": None}},
            {"longDescription": {"$eq": ""}}
        ]
    })
    stats["events_without_description"] = events_without_description

    # Trouver les événements avec longDescription trop courte
    # On doit utiliser $where car MongoDB ne supporte pas directement
    # la comparaison de longueur de string dans les queries simples
    pipeline = [
        {
            "$match": {
                "longDescription": {"$exists": True, "$nin": [None, ""]}
            }
        },
        {
            "$project": {
                "_id": 1,
                "uid": 1,
                "title": 1,
                "longDescription": 1,
                "descriptionLength": {"$strLenCP": "$longDescription"}
            }
        },
        {
            "$match": {
                "descriptionLength": {"$lt": min_length}
            }
        }
    ]

    events_with_short_description = list(collection.aggregate(pipeline))
    stats["events_with_short_description"] = len(events_with_short_description)

    # Total des événements à supprimer
    stats["events_to_delete"] = (
        stats["events_without_description"] +
        stats["events_with_short_description"]
    )

    logger.info(f"📊 Événements sans longDescription: "
                f"{stats['events_without_description']}")
    logger.info(f"📊 Événements avec longDescription < {min_length} caractères: "
                f"{stats['events_with_short_description']}")
    logger.info(f"🗑️  Total d'événements à supprimer: {stats['events_to_delete']}")

    if stats["events_to_delete"] == 0:
        logger.info("✅ Aucun événement à nettoyer")
        return stats

    # Afficher quelques exemples
    if events_with_short_description:
        logger.info("\n📝 Exemples d'événements avec description courte:")
        for i, event in enumerate(events_with_short_description[:3], 1):
            title = event.get("title", "Sans titre")
            desc_len = event.get("descriptionLength", 0)
            logger.info(f"  {i}. {title} ({desc_len} caractères)")

    if dry_run:
        logger.info("\n🔍 MODE DRY-RUN: Aucune suppression effectuée")
        return stats

    # Supprimer les événements
    logger.info("\nSuppression des événements en cours...")

    # Supprimer ceux sans description
    if stats["events_without_description"] > 0:
        result1 = collection.delete_many({
            "$or": [
                {"longDescription": {"$exists": False}},
                {"longDescription": None},
                {"longDescription": ""}
            ]
        })
        logger.info(f"  - {result1.deleted_count} événements sans description "
                    "supprimés")

    # Supprimer ceux avec description trop courte
    if stats["events_with_short_description"] > 0:
        ids_to_delete = [event["_id"] for event in events_with_short_description]
        result2 = collection.delete_many({"_id": {"$in": ids_to_delete}})
        logger.info(f"  - {result2.deleted_count} événements avec description "
                    f"< {min_length} caractères supprimés")

    # Compter le total supprimé
    new_total = collection.count_documents({})
    stats["events_deleted"] = stats["total_events"] - new_total

    logger.info(f"\n✅ {stats['events_deleted']} événements supprimés au total")
    logger.info(f"📊 Nombre d'événements après nettoyage: {new_total}")
    logger.info(f"📉 Réduction: {stats['events_deleted']} événements "
                f"({stats['events_deleted'] / stats['total_events'] * 100:.1f}%)")

    return stats


def main():
    """
    Point d'entrée principal du script de nettoyage.
    """
    logger.info("=" * 70)
    logger.info("NETTOYAGE DE LA COLLECTION EVENTS")
    logger.info("=" * 70)
    logger.info(f"Critère: longDescription >= {MIN_DESCRIPTION_LENGTH} caractères")
    logger.info("")

    client = None
    try:
        # Connexion à MongoDB
        client, collection = get_mongodb_connection()

        # Exécuter le nettoyage
        stats = clean_events(
            collection, min_length=MIN_DESCRIPTION_LENGTH, dry_run=False
        )

        # Afficher le résumé
        logger.info("\n" + "=" * 70)
        logger.info("RÉSUMÉ DU NETTOYAGE")
        logger.info("=" * 70)
        logger.info(f"Events avant nettoyage: {stats['total_events']}")
        logger.info(
            f"Events sans description: {stats['events_without_description']}"
        )
        logger.info(
            f"Events avec description courte: "
            f"{stats['events_with_short_description']}"
        )
        logger.info(f"Events supprimés: {stats['events_deleted']}")
        logger.info(
            f"Events après nettoyage: "
            f"{stats['total_events'] - stats['events_deleted']}"
        )
        logger.info("=" * 70)

        if stats['events_deleted'] > 0:
            logger.info("✅ Nettoyage terminé avec succès")
        else:
            logger.info("✅ Aucun événement à nettoyer")

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()
            logger.info("Connexion MongoDB fermée")


if __name__ == "__main__":
    main()
