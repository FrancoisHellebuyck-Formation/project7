import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

import logging
from pymongo import MongoClient, UpdateOne

load_dotenv()
OA_BASE_URL = os.getenv("OA_BASE_URL")  # Base URL de l'API OpenAgenda
API_KEY = os.getenv("OA_API_KEY")
REQUEST_AGENDA_REGION = os.getenv("OA_REGION")  # Région ciblée
AGENDAS_ENDPOINT = os.getenv("OA_AGENDAS_ENDPOINT")  # Endpoint pour les agendas
OA_PAGE_SIZE = int(os.getenv("OA_PAGE_SIZE", "100"))  # Taille de page par défaut


def get_default_updated_date() -> str:
    """
    Calcule la date par défaut pour updatedAt.gte (aujourd'hui - 1 an).

    Returns:
        str: Date au format ISO 8601 (ex: "2024-11-03T00:00:00.000Z")
    """
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    return one_year_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# Date de mise à jour minimale pour filtrer les agendas
# Par défaut: date du jour - 1 an
# Peut être surchargée via la variable d'environnement OA_AGENDAS_UPDATED_AT_GTE
OA_AGENDAS_UPDATED_AT_GTE = os.getenv(
    "OA_AGENDAS_UPDATED_AT_GTE",
    get_default_updated_date()
)

# --- Connexion à MongoDB ---
# Assurez-vous que votre conteneur Docker MongoDB est en cours d'exécution
client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGODB_DB_NAME", "mydatabase")]  # Le nom de la DB
agendas_collection = db[
    os.getenv("MONGODB_COLLECTION_NAME_AGENDAS", "agendas")
]  # collection

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,  # Niveau de logging par défaut (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paramètres de la requête pour cibler les agendas
parametres_agendas = {
    "key": API_KEY,
    "official": 1,  # Agendas officiels uniquement
    "search": REQUEST_AGENDA_REGION,  # Région ciblée
    "size": OA_PAGE_SIZE,  # Nombre maximum d'agendas à récupérer par requête
    "updatedAt.gte": OA_AGENDAS_UPDATED_AT_GTE,  # Date de mise à jour minimale
    "sort": "createdAt.desc",
}

# Initialisation pour la pagination
current_params = parametres_agendas.copy()

nb_agendas_total = 0
nb_agendas_inserted = 0
nb_agendas_updated = 0

# Log de la date de filtrage utilisée
logger.info("=" * 70)
logger.info("RÉCUPÉRATION DES AGENDAS DEPUIS OPENAGENDA API")
logger.info("=" * 70)
logger.info(f"Région ciblée: {REQUEST_AGENDA_REGION}")
logger.info(f"Date de mise à jour minimale: {OA_AGENDAS_UPDATED_AT_GTE}")
logger.info("=" * 70)

try:
    while True:
        try:
            # 1. Construction de l'URL et envoi de la requête GET
            print_params = {k: v for k, v in current_params.items() if k != "key"}
            logger.debug(f"Requête avec les paramètres : {print_params}")
            reponse = requests.get(
                OA_BASE_URL + AGENDAS_ENDPOINT, params=current_params
            )
            # print(f"URL de la requête : {reponse.url}")
            # 2. Vérification du statut HTTP
            reponse.raise_for_status()  # Lève une exception pour les codes 4xx/5xx

            # 3. Extraction des données JSON
            data_agendas = reponse.json()
            agendas_to_insert = []
            for agenda in data_agendas.get("agendas", []):
                # logger.debug(f"Agenda trouvé : uid={agenda.get('uid')}")
                agendas_to_insert.append(agenda)

            # 4. Insertion des données dans MongoDB
            if agendas_to_insert:
                # Utilisation de bulk_write pour de meilleures performances
                operations = [
                    UpdateOne({"uid": agenda["uid"]}, {"$set": agenda}, upsert=True)
                    for agenda in agendas_to_insert
                ]
                result = agendas_collection.bulk_write(operations)
                nb_agendas_total += len(agendas_to_insert)
                nb_agendas_inserted += result.upserted_count
                nb_agendas_updated += result.modified_count
                logger.info(
                    f"---> {len(agendas_to_insert)} agendas traités : {result.upserted_count} insérés, {result.modified_count} modifiés dans '{agendas_collection.name}'."
                )

            logger.info(f"Statut HTTP : {reponse.status_code}")
            # Gestion de la pagination
            if data_agendas.get("after"):
                # Pour la page suivante, on ne garde que le curseur et la clé API
                current_params["after[]"] = data_agendas["after"]
            else:
                logger.info("--------------------------------------------------")
                logger.info("Fin de la récupération des agendas.")
                logger.info(f"Nombre total d'agendas récupérés : {nb_agendas_total}")
                logger.info(f"Nombre d'agendas insérés : {nb_agendas_inserted}")
                logger.info(f"Nombre d'agendas mis à jour : {nb_agendas_updated}")
                logger.info("--------------------------------------------------")
                logger.info(
                    f"data_agendas: {data_agendas}"
                )  # Utilisation de debug pour les données brutes
                logger.info("--------------------------------------------------")
                break  # Sortir de la boucle si aucun agenda n'est trouvé

        except Exception as err:
            logger.error(
                f"Une autre erreur est survenue : {err}", exc_info=True
            )  # exc_info=True pour inclure la traceback
            break
finally:
    # 5. Fermeture de la connexion
    client.close()  # type: ignore
    logger.info("Connexion à MongoDB fermée.")
