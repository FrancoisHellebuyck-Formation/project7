import requests
import os
from dotenv import load_dotenv

from pymongo import MongoClient

load_dotenv()
API_KEY = os.getenv("OA_API_KEY")
BASE_URL = os.getenv("OA_BASE_URL")
REQUEST_AGENDA = (
    "https://api.openagenda.com/v2/agendas"  # URL de base de l'API OpenAgenda
)
REQUEST_AGENDA_REGION = os.getenv("OA_REGION")  # Région ciblée

# --- Connexion à MongoDB ---
# Assurez-vous que votre conteneur Docker MongoDB est en cours d'exécution
client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGODB_DATABASE", "mydatabase")]  # Le nom de la DB
agendas_collection = db[
    os.getenv("MONGODB_COLLECTION_NAME_AGENDAS", "agendas")
]  # collection

# Paramètres de la requête pour cibler les agendas
parametres_agendas = {
    "key": API_KEY,
    "official": 1,  # Agendas officiels uniquement
    "search": REQUEST_AGENDA_REGION,  # Région ciblée
    "size": 100,  # Nombre maximum d'agendas à récupérer par requête
    "updatedAt.gte": "2025-01-01T12:00:00.000Z",
}
agendas_endpoint = f"{BASE_URL}/agendas"
try:
    while True:
        try:
            # 1. Construction de l'URL et envoi de la requête GET
            reponse = requests.get(agendas_endpoint, params=parametres_agendas)

            # 2. Vérification du statut HTTP
            reponse.raise_for_status()  # Lève une exception pour les codes 4xx/5xx

            # 3. Extraction des données JSON
            data_agendas = reponse.json()
            agendas_to_insert = []
            for agenda in data_agendas.get("agendas", []):
                print(f"Agenda trouvé : uid={agenda.get('uid')}")
                agendas_to_insert.append(agenda)

            # 4. Insertion des données dans MongoDB
            if agendas_to_insert:
                # On peut utiliser update_one avec upsert=True pour éviter les doublons si le script est lancé plusieurs fois
                for agenda in agendas_to_insert:
                    agendas_collection.update_one(
                        {"uid": agenda["uid"]}, {"$set": agenda}, upsert=True
                    )
                print(
                    f"\n{len(agendas_to_insert)} agendas insérés ou mis à jour dans la collection '{agendas_collection.name}'."
                )

            print(f"Statut HTTP : {reponse.status_code}")
            # Gestion de la pagination
            if data_agendas.get("after"):
                parametres_agendas["after[]"] = data_agendas["after"]  # [0]
                # parametres_agendas["after[]"] = data_agendas["after"][1]
            else:
                break  # Sortir de la boucle si pas de page suivante

        except requests.exceptions.HTTPError as err:
            print(f"Erreur HTTP : {err}")
            break
        except Exception as err:
            print(f"Une autre erreur est survenue : {err}")
            break
finally:
    # 5. Fermeture de la connexion
    client.close()
    print("Connexion à MongoDB fermée.")
