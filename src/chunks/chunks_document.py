# %%
# add this import for running in jupyter notebook
import nest_asyncio

nest_asyncio.apply()

# %% Importation des bibliothèques nécessaires
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pymongo import MongoClient

from dotenv import load_dotenv
import os

# %% Chargement des données depuis MongoDB
load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGODB_DB_NAME", "mydatabase")]  # Le nom de la DB
events_collection = db[
    os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")
]  # collection events


# %% Découpage des documents en chunks
def format_event_content(doc):
    """Formatte le document MongoDB en texte structuré"""

    # Construction du contenu textuel
    content_parts = []

    # Titre et informations principales
    content_parts.append(f"Titre: {doc.get('title', 'Sans titre')}")
    content_parts.append(f"Date: {doc.get('dateRange', '')}")
    content_parts.append(f"Conditions: {doc.get('conditions', '')}")

    # Description
    if doc.get("description"):
        content_parts.append(f"\nDescription: {doc['description']}")

    if doc.get("longDescription"):
        content_parts.append(f"\nDescription détaillée: {doc['longDescription']}")

    # Localisation
    location = doc.get("location", {})
    if location:
        loc_text = f"\nLieu: {location.get('name', '')}"
        loc_text += f"\nAdresse: {location.get('address', '')}"
        loc_text += f"\nVille: {location.get('city', '')}"
        loc_text += f"\nRégion: {location.get('region', '')}"
        content_parts.append(loc_text)

    # Mots-clés
    keywords = doc.get("keywords", [])
    if keywords:
        content_parts.append(f"\nMots-clés: {', '.join(keywords)}")

    # Mode d'accès
    attendance = doc.get("attendanceMode", {})
    if attendance:
        content_parts.append(f"\nMode: {attendance.get('label', '')}")

    return "\n".join(content_parts)


def extract_metadata(doc):
    """Extrait les métadonnées importantes"""
    location = doc.get("location", {})

    metadata = {
        "event_id": str(doc.get("_id")),
        "uid": doc.get("uid"),
        "title": doc.get("title", ""),
        "city": location.get("city", ""),
        "department": location.get("department", ""),
        "region": location.get("region", ""),
        "postal_code": location.get("postalCode", ""),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "date_debut": doc.get("firstTiming", {}).get("begin"),
        "date_fin": doc.get("lastTiming", {}).get("end"),
        "keywords": doc.get("keywords", []),
        "conditions": doc.get("conditions", ""),
        "status": doc.get("status", {}).get("label", ""),
    }

    # Nettoyer les valeurs None
    return {k: v for k, v in metadata.items() if v is not None}


# %%
# Récupération et transformation des documents
documents = []

for doc in events_collection.find():
    # Formater le contenu
    page_content = format_event_content(doc)

    # Extraire les métadonnées
    metadata = extract_metadata(doc)

    # Créer le Document LangChain
    documents.append(Document(page_content=page_content, metadata=metadata))

print(f"Nombre de documents chargés : {len(documents)}")


# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"Nombre de chunks créés : {len(chunks)}")

# Exemple d'affichage
if chunks:
    print("\n--- Premier chunk ---")
    print(f"Contenu:\n{chunks[0].page_content}")
    print(f"\nMétadonnées:\n{chunks[0].metadata}")
# %%
