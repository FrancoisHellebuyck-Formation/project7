"""
Tests unitaires pour le module chunks (chunks_document.py).

Ce module teste le découpage de documents d'événements en chunks.
"""

import os
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_environment():
    """Configure l'environnement de test."""
    env_vars = {
        "MONGODB_URI": "mongodb://localhost:27017/",
        "MONGODB_DB_NAME": "test_db",
        "MONGODB_COLLECTION_NAME_EVENTS": "test_events",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "100",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.fixture
def sample_mongodb_doc():
    """Document MongoDB de test."""
    return {
        "_id": "507f1f77bcf86cd799439011",
        "uid": "event-123",
        "title": "Concert de Jazz",
        "dateRange": "2024-12-15 19:00 - 21:00",
        "conditions": "Entrée gratuite",
        "description": "Concert de jazz gratuit dans le parc",
        "longDescription": "Description détaillée du concert avec programme complet",
        "location": {
            "name": "Parc des expositions",
            "address": "123 Rue de la Musique",
            "city": "Paris",
            "region": "Île-de-France",
            "department": "Paris",
            "postalCode": "75001",
            "latitude": 48.8566,
            "longitude": 2.3522
        },
        "keywords": ["jazz", "musique", "concert"],
        "attendanceMode": {"label": "Présentiel"},
        "firstTiming": {"begin": "2024-12-15T19:00:00"},
        "lastTiming": {"end": "2024-12-15T21:00:00"},
        "status": {"label": "Confirmé"}
    }


@pytest.mark.unit
def test_get_mongodb_connection(mock_environment):
    """Teste la connexion à MongoDB."""
    with patch("chunks.chunks_document.MongoClient") as mock_mongo:
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_mongo.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection

        from chunks.chunks_document import get_mongodb_connection

        client, collection = get_mongodb_connection()

        assert client == mock_client
        assert collection == mock_collection
        mock_mongo.assert_called_once_with("mongodb://localhost:27017/")


@pytest.mark.unit
def test_format_event_content(sample_mongodb_doc):
    """Teste le formatage du contenu d'un événement."""
    from chunks.chunks_document import format_event_content

    content = format_event_content(sample_mongodb_doc)

    # Vérifier que les éléments clés sont présents
    assert "Concert de Jazz" in content
    assert "2024-12-15 19:00 - 21:00" in content
    assert "Entrée gratuite" in content
    assert "Concert de jazz gratuit" in content
    assert "Parc des expositions" in content
    assert "Paris" in content
    assert "Île-de-France" in content
    assert "jazz, musique, concert" in content


@pytest.mark.unit
def test_format_event_content_minimal():
    """Teste le formatage avec données minimales."""
    from chunks.chunks_document import format_event_content

    minimal_doc = {
        "title": "Événement simple"
    }

    content = format_event_content(minimal_doc)

    assert "Événement simple" in content
    assert "Sans titre" not in content


@pytest.mark.unit
def test_format_event_content_missing_title():
    """Teste le formatage sans titre."""
    from chunks.chunks_document import format_event_content

    doc_no_title = {}

    content = format_event_content(doc_no_title)

    assert "Sans titre" in content


@pytest.mark.unit
def test_extract_metadata(sample_mongodb_doc):
    """Teste l'extraction des métadonnées."""
    from chunks.chunks_document import extract_metadata

    metadata = extract_metadata(sample_mongodb_doc)

    assert metadata["uid"] == "event-123"
    assert metadata["title"] == "Concert de Jazz"
    assert metadata["city"] == "Paris"
    assert metadata["region"] == "Île-de-France"
    assert metadata["department"] == "Paris"
    assert metadata["postal_code"] == "75001"
    assert metadata["latitude"] == 48.8566
    assert metadata["longitude"] == 2.3522
    assert metadata["keywords"] == ["jazz", "musique", "concert"]
    assert metadata["status"] == "Confirmé"


@pytest.mark.unit
def test_extract_metadata_removes_none_values():
    """Teste que les valeurs None sont exclues des métadonnées."""
    from chunks.chunks_document import extract_metadata

    doc_with_none = {
        "_id": "123",
        "title": "Test",
        "location": {},
        "firstTiming": {},
        "lastTiming": {}
    }

    metadata = extract_metadata(doc_with_none)

    # Vérifier que les clés avec None ne sont pas présentes
    assert "latitude" not in metadata
    assert "longitude" not in metadata
    assert "date_debut" not in metadata
    assert "date_fin" not in metadata


@pytest.mark.unit
def test_load_documents_from_mongodb(sample_mongodb_doc):
    """Teste le chargement de documents depuis MongoDB."""
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = [sample_mongodb_doc]
    mock_collection.find.return_value = mock_cursor

    from chunks.chunks_document import load_documents_from_mongodb

    documents = load_documents_from_mongodb(mock_collection)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "Concert de Jazz" in documents[0].page_content
    assert documents[0].metadata["uid"] == "event-123"
    assert documents[0].metadata["city"] == "Paris"


@pytest.mark.unit
def test_load_documents_with_query(sample_mongodb_doc):
    """Teste le chargement avec une query spécifique."""
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = [sample_mongodb_doc]
    mock_collection.find.return_value = mock_cursor

    from chunks.chunks_document import load_documents_from_mongodb

    query = {"city": "Paris"}
    documents = load_documents_from_mongodb(mock_collection, query=query)

    mock_collection.find.assert_called_once_with(query)
    assert len(documents) == 1


@pytest.mark.unit
def test_load_documents_with_limit(sample_mongodb_doc):
    """Teste le chargement avec une limite."""
    mock_collection = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.__iter__.return_value = [sample_mongodb_doc]
    mock_collection.find.return_value = mock_cursor

    from chunks.chunks_document import load_documents_from_mongodb

    documents = load_documents_from_mongodb(mock_collection, limit=10)

    mock_cursor.limit.assert_called_once_with(10)
    assert len(documents) == 1


@pytest.mark.unit
def test_create_text_splitter():
    """Teste la création du text splitter."""
    from chunks.chunks_document import create_text_splitter

    splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)

    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 50


@pytest.mark.unit
def test_create_text_splitter_default_values():
    """Teste la création avec valeurs par défaut."""
    from chunks.chunks_document import create_text_splitter

    splitter = create_text_splitter()

    assert splitter._chunk_size == 1500
    assert splitter._chunk_overlap == 200


@pytest.mark.unit
def test_get_chunk_parameters(mock_environment):
    """Teste la récupération des paramètres de chunking."""
    from chunks.chunks_document import get_chunk_parameters

    chunk_size, chunk_overlap = get_chunk_parameters()

    assert chunk_size == 500
    assert chunk_overlap == 100


@pytest.mark.unit
def test_get_chunk_parameters_defaults():
    """Teste les valeurs par défaut des paramètres."""
    with patch.dict(os.environ, {}, clear=True):
        from chunks.chunks_document import get_chunk_parameters

        chunk_size, chunk_overlap = get_chunk_parameters()

        assert chunk_size == 500
        assert chunk_overlap == 100


@pytest.mark.unit
def test_split_documents_into_chunks():
    """Teste le découpage de documents en chunks."""
    from chunks.chunks_document import split_documents_into_chunks

    # Créer un document long
    long_content = "Lorem ipsum dolor sit amet. " * 100  # ~2800 caractères
    doc = Document(
        page_content=long_content,
        metadata={"title": "Test", "city": "Paris"}
    )

    chunks = split_documents_into_chunks([doc], chunk_size=500, chunk_overlap=100)

    # Devrait créer plusieurs chunks
    assert len(chunks) > 1

    # Vérifier que les métadonnées sont conservées
    for chunk in chunks:
        assert chunk.metadata["title"] == "Test"
        assert chunk.metadata["city"] == "Paris"


@pytest.mark.unit
def test_split_documents_short_document():
    """Teste le découpage d'un document court (pas de split)."""
    from chunks.chunks_document import split_documents_into_chunks

    short_content = "Document court"
    doc = Document(
        page_content=short_content,
        metadata={"title": "Test"}
    )

    chunks = split_documents_into_chunks([doc], chunk_size=500, chunk_overlap=100)

    # Un seul chunk car document court
    assert len(chunks) == 1
    assert chunks[0].page_content == short_content


@pytest.mark.unit
def test_split_documents_multiple_documents():
    """Teste le découpage de plusieurs documents."""
    from chunks.chunks_document import split_documents_into_chunks

    docs = [
        Document(page_content="Document 1 " * 100, metadata={"id": "1"}),
        Document(page_content="Document 2 " * 100, metadata={"id": "2"}),
    ]

    chunks = split_documents_into_chunks(docs, chunk_size=500, chunk_overlap=100)

    # Devrait avoir des chunks pour les deux documents
    assert len(chunks) > 2

    # Vérifier que chaque document original a généré des chunks
    doc1_chunks = [c for c in chunks if c.metadata.get("id") == "1"]
    doc2_chunks = [c for c in chunks if c.metadata.get("id") == "2"]

    assert len(doc1_chunks) > 0
    assert len(doc2_chunks) > 0


@pytest.mark.unit
def test_split_documents_with_env_defaults():
    """Teste le découpage avec les valeurs par défaut du .env."""
    from chunks.chunks_document import split_documents_into_chunks

    doc = Document(page_content="Test " * 200, metadata={})

    # Sans spécifier chunk_size/overlap, utilise les valeurs du .env
    chunks = split_documents_into_chunks([doc])

    # Devrait avoir créé des chunks
    assert len(chunks) > 0
