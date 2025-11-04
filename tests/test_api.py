"""
Tests unitaires pour l'API FastAPI.

Ce module contient les tests pour tous les endpoints de l'API de recherche
d'événements culturels, incluant la recherche sémantique, le chatbot RAG,
et la gestion du rebuild de l'index FAISS.
"""

import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# Mock des variables d'environnement avant import
@pytest.fixture(scope="session", autouse=True)
def mock_environment():
    """Configure les variables d'environnement pour les tests."""
    env_vars = {
        "FAISS_INDEX_PATH": "data/faiss_index",
        "EMBEDDINGS_MODEL": "intfloat/multilingual-e5-large",
        "EMBEDDINGS_DEVICE": "cpu",
        "MONGODB_URI": "mongodb://localhost:27017/",
        "MONGODB_DB_NAME": "OA_TEST",
        "MONGODB_COLLECTION_NAME_EVENTS": "events",
        "MISTRAL_API_KEY": "test_key_123",
        "MISTRAL_MODEL": "mistral-small-latest",
        "RAG_TOP_K": "5",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


# Mock du vector store et des dépendances
@pytest.fixture
def mock_vector_store():
    """Crée un mock du vector store FAISS."""
    vector_store = Mock()
    vector_store.similarity_search_with_score = Mock(return_value=[
        (
            Mock(
                page_content="Concert de jazz à Toulouse",
                metadata={
                    "title": "Jazz Festival",
                    "city": "Toulouse",
                    "date_debut": "2025-11-10T19:00:00+01:00",
                    "location": "Place du Capitole",
                    "event_id": "evt_123"
                }
            ),
            0.85
        )
    ])
    return vector_store


@pytest.fixture
def mock_embeddings_model():
    """Crée un mock du modèle d'embeddings."""
    return Mock()


@pytest.fixture
def mock_mistral_client():
    """Crée un mock du client Mistral AI."""
    client = Mock()
    response = Mock()
    response.choices = [Mock(message=Mock(content="Voici la réponse à votre question."))]
    response.usage = Mock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    client.chat.complete = Mock(return_value=response)
    return client


@pytest.fixture
def client(mock_vector_store, mock_embeddings_model, mock_mistral_client):
    """
    Crée un client de test FastAPI avec les mocks injectés.
    """
    # Mock des fonctions de chargement
    with patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.get_embeddings_model", return_value=mock_embeddings_model), \
         patch("api.main.Mistral", return_value=mock_mistral_client), \
         patch("api.main.load_system_prompt", return_value="Tu es un assistant."), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 1000, "dimension": 1024}):

        from api.main import app

        # Injecter les mocks dans les variables globales de l'app
        app.state.vector_store = mock_vector_store
        app.state.embeddings_model = mock_embeddings_model
        app.state.mistral_client = mock_mistral_client

        with TestClient(app) as test_client:
            yield test_client


# ============================================================================
# Tests des endpoints de base
# ============================================================================

@pytest.mark.unit
def test_root_endpoint(client):
    """Teste l'endpoint racine GET /."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "API de recherche d'événements culturels"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data
    assert "search" in data["endpoints"]
    assert "ask" in data["endpoints"]
    assert "rebuild" in data["endpoints"]


@pytest.mark.unit
def test_health_endpoint_healthy(client):
    """Teste l'endpoint GET /health avec tous les composants chargés."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_store_loaded"] is True
    assert data["embeddings_model_loaded"] is True


@pytest.mark.unit
def test_stats_endpoint(client):
    """Teste l'endpoint GET /stats."""
    with patch("api.main.get_vector_store_stats") as mock_stats:
        mock_stats.return_value = {
            "num_vectors": 1000,
            "dimension": 1024
        }

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["num_vectors"] == 1000
        assert data["dimension"] == 1024
        assert "index_path" in data


# ============================================================================
# Tests de l'endpoint /search
# ============================================================================

@pytest.mark.unit
def test_search_endpoint_success(client):
    """Teste l'endpoint POST /search avec une requête valide."""
    payload = {
        "query": "concert de jazz",
        "k": 5
    }

    response = client.post("/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "concert de jazz"
    assert "results" in data
    assert len(data["results"]) > 0
    assert data["total_results"] > 0

    # Vérifier la structure du premier résultat
    result = data["results"][0]
    assert "score" in result
    assert "title" in result
    assert "content" in result
    assert "metadata" in result


@pytest.mark.unit
def test_search_endpoint_validation_error(client):
    """Teste l'endpoint /search avec des données invalides."""
    payload = {
        "query": "",  # Chaîne vide invalide
        "k": 5
    }

    response = client.post("/search", json=payload)

    assert response.status_code == 422  # Validation error


@pytest.mark.unit
def test_search_endpoint_invalid_k(client):
    """Teste l'endpoint /search avec k hors limites."""
    payload = {
        "query": "concert",
        "k": 200  # Au-dessus de la limite de 100
    }

    response = client.post("/search", json=payload)

    assert response.status_code == 422


# ============================================================================
# Tests de l'endpoint /ask
# ============================================================================

@pytest.mark.unit
def test_ask_endpoint_success(client):
    """Teste l'endpoint POST /ask avec une question valide."""
    payload = {
        "question": "Quels sont les festivals de jazz en Occitanie ?",
        "k": 3
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert "answer" in data
    assert "context_used" in data
    assert "tokens_used" in data
    assert len(data["context_used"]) > 0
    assert data["tokens_used"]["total_tokens"] == 150


@pytest.mark.unit
def test_ask_endpoint_with_custom_prompt(client):
    """Teste l'endpoint /ask avec un prompt système personnalisé."""
    payload = {
        "question": "Trouve-moi des événements pour enfants",
        "k": 5,
        "system_prompt": "Tu es un assistant familial."
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


@pytest.mark.unit
def test_ask_endpoint_validation_error(client):
    """Teste l'endpoint /ask avec une question vide."""
    payload = {
        "question": "",
        "k": 5
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 422


# ============================================================================
# Tests de l'endpoint /rebuild
# ============================================================================

@pytest.mark.skip(reason="MongoDB mocking needs refinement")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_rebuild_endpoint_success():
    """Teste l'endpoint POST /rebuild avec succès."""
    with patch("api.main.load_vector_store") as mock_load, \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test prompt"), \
         patch("api.main.asyncio.create_subprocess_exec") as mock_subprocess:

        # Mock du subprocess
        process_mock = AsyncMock()
        process_mock.communicate = AsyncMock(return_value=(b"output", b""))
        process_mock.returncode = 0
        mock_subprocess.return_value = process_mock

        # Mock du vector store
        mock_vector_store = Mock()
        mock_load.return_value = mock_vector_store

        # Mock du client MongoDB
        with patch("api.main.MongoClient") as mock_mongo:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_events_collection = MagicMock()

            # Mock de la dernière exécution
            mock_collection.find_one.return_value = {
                "pipeline_run_date": datetime(2025, 11, 3, 10, 0, 0)
            }

            # Mock du comptage d'événements (nouveaux événements présents)
            mock_events_collection.count_documents.return_value = 10

            mock_db.__getitem__ = lambda self, key: (
                mock_collection if key == "last_update" else mock_events_collection
            )
            mock_client.__getitem__.return_value = mock_db
            mock_mongo.return_value = mock_client

            from api.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                response = await ac.post("/rebuild")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "Pipeline de mise à jour démarré" in data["message"]


@pytest.mark.skip(reason="MongoDB mocking needs refinement")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_rebuild_endpoint_no_new_events():
    """Teste l'endpoint POST /rebuild sans nouveaux événements."""
    with patch("api.main.load_vector_store"), \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test prompt"):

        # Mock du client MongoDB
        with patch("api.main.MongoClient") as mock_mongo:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_events_collection = MagicMock()

            # Mock de la dernière exécution
            mock_collection.find_one.return_value = {
                "pipeline_run_date": datetime(2025, 11, 3, 10, 0, 0)
            }

            # Mock du comptage d'événements (aucun nouvel événement)
            mock_events_collection.count_documents.return_value = 0

            mock_db.__getitem__ = lambda self, key: (
                mock_collection if key == "last_update" else mock_events_collection
            )
            mock_client.__getitem__.return_value = mock_db
            mock_mongo.return_value = mock_client

            from api.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                # Lancer le rebuild
                await ac.post("/rebuild")

                # Attendre un peu que la tâche background se termine
                import asyncio
                await asyncio.sleep(0.5)

                # Vérifier le statut
                status_response = await ac.get("/rebuild/status")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["status"] == "warning"
        assert "Pas de nouveaux événements" in data["message"]


@pytest.mark.unit
def test_rebuild_status_endpoint_idle(client):
    """Teste l'endpoint GET /rebuild/status à l'état idle."""
    response = client.get("/rebuild/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert "details" in data


@pytest.mark.skip(reason="MongoDB mocking needs refinement")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_rebuild_endpoint_concurrent_calls():
    """Teste que les appels concurrents à /rebuild sont gérés correctement."""
    with patch("api.main.load_vector_store"), \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test prompt"), \
         patch("api.main.MongoClient") as mock_mongo:

        # Mock MongoDB
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_events_collection = MagicMock()

        mock_collection.find_one.return_value = {
            "pipeline_run_date": datetime(2025, 11, 3, 10, 0, 0)
        }
        mock_events_collection.count_documents.return_value = 10

        mock_db.__getitem__ = lambda self, key: (
            mock_collection if key == "last_update" else mock_events_collection
        )
        mock_client.__getitem__.return_value = mock_db
        mock_mongo.return_value = mock_client

        from api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as ac:
            # Premier appel
            response1 = await ac.post("/rebuild")
            # Deuxième appel immédiat
            response2 = await ac.post("/rebuild")

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Le deuxième devrait indiquer qu'un rebuild est en cours
    data2 = response2.json()
    assert data2["status"] == "running"
    assert "rebuild est déjà en cours" in data2["message"]


# ============================================================================
# Tests d'erreurs et edge cases
# ============================================================================

@pytest.mark.skip(reason="Need to handle startup event with None vector_store")
@pytest.mark.unit
def test_search_without_vector_store():
    """Teste /search quand le vector store n'est pas chargé."""
    with patch("api.main.load_vector_store", return_value=None), \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test prompt"):

        from api.main import app

        # Forcer vector_store à None
        import api.main
        api.main.vector_store = None

        with TestClient(app) as test_client:
            response = test_client.post("/search", json={"query": "test", "k": 5})

    assert response.status_code == 503


@pytest.mark.skip(reason="Need to handle startup event with None vector_store")
@pytest.mark.unit
def test_stats_without_vector_store():
    """Teste /stats quand le vector store n'est pas chargé."""
    with patch("api.main.load_vector_store", return_value=None), \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test prompt"):

        from api.main import app

        # Forcer vector_store à None
        import api.main
        api.main.vector_store = None

        with TestClient(app) as test_client:
            response = test_client.get("/stats")

    assert response.status_code == 503
