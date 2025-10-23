import requests
import os
from dotenv import load_dotenv

import logging
from pymongo import MongoClient, UpdateOne

load_dotenv()
API_KEY = os.getenv("OA_API_KEY")
OA_BASE_URL = os.getenv("OA_BASE_URL")  # Base URL de l'API OpenAgenda
REQUEST_AGENDA_REGION = os.getenv("OA_REGION")  # Région ciblée
OA_EVENTS_PATH_SUFFIX = os.getenv(
    "OA_EVENTS_PATH_SUFFIX", "/events"
)  # Suffixe de chemin pour les événements
OA_PAGE_SIZE = int(os.getenv("OA_PAGE_SIZE", "100"))  # Taille de page par défaut
OA_REGION = os.getenv("OA_REGION")  # Région ciblée
# --- Connexion à MongoDB ---
# Assurez-vous que votre conteneur Docker MongoDB est en cours d'exécution
client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGODB_DB_NAME", "mydatabase")]  # Le nom de la DB
agendas_collection = db[
    os.getenv("MONGODB_COLLECTION_NAME_AGENDAS", "agendas")
]  # collection agendas

events_collection = db[
    os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")
]  # collection events


# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,  # Niveau de logging par défaut (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paramètres de la requête pour cibler les events
fields_events = [
    "uid",
    "keywords",
    "attendanceMode",
    "dateRange",
    "description",
    "title",
    "status",
    "onlineAccessLink",
    "lastTiming",
    "firstTiming",
    "nextTiming",
    "location.city",
    "location.access",
    "location.postalCode",
    "location.latitude",
    "location.description",
    "location.countryCode",
    "location.links",
    "location.department",
    "location.email",
    "location.longitude",
    "location.website",
    "location.address",
    "location.tags",
    "location.insee",
    "location.phone",
    "location.district",
    "location.name",
    "location.region",
    "age",
    "longDescription",
    "conditions",
    "registration",
    "accessibility",
    "timings",
    "locationUid",
    "links",
    "timezone",
    "state",
    "createdAt",
    "updatedAt",
]

parametres_events = {
    "key": API_KEY,
    "relative[]": "upcoming",
    "monolingual": "fr",  # Langue des événements
    "includeLabels": 1,
    "detailed": 0,  # Niveau de détail des événements
    "size": 100,  # Nombre maximum d'events à récupérer par requête
    "sort": "createdAt.desc",
    "region": OA_REGION,
    "includeFields[]": fields_events,
}

# Initialisation pour la pagination
current_params = parametres_events.copy()

nb_events_total = 0
nb_events_inserted = 0
nb_events_updated = 0

try:
    if not OA_BASE_URL:
        logger.error(
            "OA_BASE_URL n'est pas défini dans les variables d'environnement. Impossible de récupérer les événements."
        )
        exit()

    # 1. Récupérer tous les agendas de la base de données
    logger.info(
        "Récupération des agendas depuis MongoDB pour trouver leurs événements..."
    )
    # On ne récupère que l'UID de l'agenda
    agendas_cursor = agendas_collection.find({}, {"uid": 1, "_id": 0})
    agenda_uids = [agenda["uid"] for agenda in agendas_cursor]
    logger.info(f"{len(agenda_uids)} agendas trouvés dans la base de données.")

    if not agenda_uids:
        logger.warning(
            "Aucun agenda trouvé dans la base de données. Veuillez exécuter 'get_corpus_agendas.py' d'abord."
        )
        exit()

    # 2. Pour chaque agenda, récupérer ses événements
    for agenda_uid in agenda_uids:
        logger.info(f"--- Traitement de l'agenda UID : {agenda_uid} ---")

        # Construction de l'endpoint spécifique pour les événements de cet agenda
        agenda_events_endpoint = (
            f"{OA_BASE_URL}/agendas/{agenda_uid}{OA_EVENTS_PATH_SUFFIX}"
        )

        # Réinitialisation des paramètres pour chaque agenda pour gérer la pagination indépendamment
        current_params = parametres_events.copy()
        current_params["key"] = (
            API_KEY  # S'assurer que la clé API est toujours présente
        )
        current_params["size"] = OA_PAGE_SIZE  # Utiliser la taille de page définie

        agenda_events_count = 0

        while True:  # Boucle de pagination pour les événements de l'agenda courant
            try:
                # 1. Construction de l'URL et envoi de la requête GET
                print_params = {k: v for k, v in current_params.items() if k != "key"}
                logger.debug(
                    f"Requête événements pour {agenda_uid} avec les paramètres : {print_params}"
                )
                reponse = requests.get(agenda_events_endpoint, params=current_params)

                # 2. Vérification du statut HTTP
                reponse.raise_for_status()  # Lève une exception pour les codes 4xx/5xx

                # 3. Extraction des données JSON
                data_events = reponse.json()
                events_to_insert = []
                for event in data_events.get("events", []):
                    # Ajout de l'UID de l'agenda à chaque événement pour faciliter les requêtes futures
                    event["agendaUid"] = agenda_uid
                    events_to_insert.append(event)

                # 4. Insertion des données dans MongoDB
                if events_to_insert:
                    # Utilisation de bulk_write pour de meilleures performances
                    # La clé unique pour un événement est son UID ET l'UID de l'agenda parent
                    operations = [
                        UpdateOne(
                            {"uid": event["uid"], "agendaUid": event["agendaUid"]},
                            {"$set": event},
                            upsert=True,
                        )
                        for event in events_to_insert
                    ]
                    result = events_collection.bulk_write(operations)
                    nb_events_total += len(events_to_insert)
                    nb_events_inserted += result.upserted_count
                    nb_events_updated += result.modified_count
                    agenda_events_count += len(events_to_insert)
                    logger.info(
                        f"---> {len(events_to_insert)} événements traités pour l'agenda {agenda_uid}: {result.upserted_count} insérés, {result.modified_count} modifiés dans '{events_collection.name}'."
                    )

                # Gestion de la pagination
                if data_events.get("after"):
                    current_params["after[]"] = data_events["after"]
                else:
                    logger.info(
                        f"Fin de la récupération des événements pour l'agenda {agenda_uid}. Total: {agenda_events_count} événements."
                    )
                    break  # Sortir de la boucle de pagination pour cet agenda

            except requests.exceptions.RequestException as req_err:
                logger.error(
                    f"Erreur de requête pour l'agenda {agenda_uid}: {req_err}",
                    exc_info=True,
                )
                break  # Passer à l'agenda suivant en cas d'erreur de requête
            except Exception as err:
                logger.error(
                    f"Une erreur inattendue est survenue lors de la récupération des événements pour l'agenda {agenda_uid}: {err}",
                    exc_info=True,
                )
                break  # Passer à l'agenda suivant en cas d'erreur inattendue

    logger.info("--------------------------------------------------")
    logger.info("Fin de la récupération de tous les événements.")
    logger.info(f"Nombre total d'événements récupérés : {nb_events_total}")
    logger.info(f"Nombre d'événements insérés : {nb_events_inserted}")
    logger.info(f"Nombre d'événements mis à jour : {nb_events_updated}")
    logger.info("--------------------------------------------------")

except Exception as err:
    logger.error(f"Une erreur globale est survenue : {err}", exc_info=True)
finally:
    # 5. Fermeture de la connexion
    client.close()  # type: ignore
    logger.info("Connexion à MongoDB fermée.")
