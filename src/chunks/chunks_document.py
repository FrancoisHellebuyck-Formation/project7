"""
Module pour le découpage (chunking) de documents d'événements MongoDB en chunks indexables.

Ce module transforme les documents d'événements stockés dans MongoDB en chunks
de texte optimisés pour la recherche sémantique et l'indexation vectorielle.
"""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
import os


def get_mongodb_connection() -> tuple[MongoClient, Collection]:
    """
    Établit une connexion à MongoDB et retourne le client et la collection des événements.

    Returns:
        tuple: (MongoClient, Collection) - Client MongoDB et collection d'événements

    Raises:
        ValueError: Si les variables d'environnement requises sont manquantes
    """
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "mydatabase")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME_EVENTS", "events")

    client = MongoClient(mongodb_uri)
    db = client[db_name]
    events_collection = db[collection_name]

    return client, events_collection


def format_event_content(doc: Dict[str, Any]) -> str:
    """
    Formatte un document MongoDB d'événement en texte structuré.

    Args:
        doc: Document MongoDB contenant les informations de l'événement

    Returns:
        str: Contenu textuel formaté et structuré de l'événement
    """
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


def extract_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrait les métadonnées importantes d'un document d'événement.

    Args:
        doc: Document MongoDB contenant les informations de l'événement

    Returns:
        dict: Dictionnaire des métadonnées filtrées (valeurs None exclues)
    """
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


def load_documents_from_mongodb(
    events_collection: Collection,
    query: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> List[Document]:
    """
    Charge les documents depuis MongoDB et les transforme en objets Document LangChain.

    Args:
        events_collection: Collection MongoDB des événements
        query: Filtre de requête MongoDB optionnel (par défaut: {})
        limit: Nombre maximum de documents à récupérer (par défaut: tous)

    Returns:
        list[Document]: Liste de documents LangChain formatés
    """
    documents = []
    query = query or {}

    cursor = events_collection.find(query)
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        # Formater le contenu
        page_content = format_event_content(doc)

        # Extraire les métadonnées
        metadata = extract_metadata(doc)

        # Créer le Document LangChain
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def create_text_splitter(
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """
    Crée un splitter de texte configuré pour découper les documents.

    Args:
        chunk_size: Taille maximale de chaque chunk en caractères (défaut: 1500)
        chunk_overlap: Nombre de caractères de chevauchement entre chunks (défaut: 200)

    Returns:
        RecursiveCharacterTextSplitter: Instance configurée du splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Découpe une liste de documents en chunks plus petits.

    Args:
        documents: Liste de documents LangChain à découper
        chunk_size: Taille maximale de chaque chunk en caractères
        chunk_overlap: Nombre de caractères de chevauchement entre chunks

    Returns:
        list[Document]: Liste de chunks (documents découpés)
    """
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def process_events_to_chunks(
    events_collection: Collection,
    query: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    verbose: bool = False
) -> List[Document]:
    """
    Pipeline complet: charge les événements depuis MongoDB et les découpe en chunks.

    Args:
        events_collection: Collection MongoDB des événements
        query: Filtre de requête MongoDB optionnel
        limit: Nombre maximum de documents à traiter
        chunk_size: Taille maximale de chaque chunk
        chunk_overlap: Chevauchement entre chunks
        verbose: Si True, affiche des informations de progression

    Returns:
        list[Document]: Liste de tous les chunks créés
    """
    # Charger les documents
    documents = load_documents_from_mongodb(events_collection, query, limit)

    if verbose:
        print(f"Nombre de documents chargés : {len(documents)}")

    # Découper en chunks
    chunks = split_documents_into_chunks(documents, chunk_size, chunk_overlap)

    if verbose:
        print(f"Nombre de chunks créés : {len(chunks)}")
        if chunks:
            print("\n--- Premier chunk ---")
            print(f"Contenu:\n{chunks[0].page_content}")
            print(f"\nMétadonnées:\n{chunks[0].metadata}")

    return chunks


def main():
    """
    Fonction principale pour exécuter le pipeline de chunking.
    Exemple d'utilisation du module.
    """
    # Connexion à MongoDB
    client, events_collection = get_mongodb_connection()

    try:
        # Traiter les événements et créer les chunks
        chunks = process_events_to_chunks(
            events_collection=events_collection,
            verbose=True
        )

        print(f"\n✓ Pipeline terminé avec succès: {len(chunks)} chunks créés")

    finally:
        # Fermer la connexion
        client.close()
        print("Connexion à MongoDB fermée.")


if __name__ == "__main__":
    main()
