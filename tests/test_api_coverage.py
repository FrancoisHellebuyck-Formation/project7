"""
Tests additionnels pour améliorer la couverture de code de l'API.

Ce module contient des tests ciblés pour atteindre 80% de couverture
en testant les branches et cas d'erreur non couverts par test_api.py.
"""

import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_environment():
    """Configure les variables d'environnement pour les tests."""
    env_vars = {
        "FAISS_INDEX_PATH": "data/faiss_index",
        "EMBEDDINGS_MODEL": "intfloat/multilingual-e5-large",
        "EMBEDDINGS_DEVICE": "cpu",
        "MONGODB_URI": "mongodb://localhost:27017/",
        "MONGODB_DB_NAME": "OA_TEST",
        "MONGODB_COLLECTION_NAME_EVENTS": "events",
        "MISTRAL_API_KEY": "test_key",
        "MISTRAL_MODEL": "mistral-small-latest",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


# ============================================================================
# Tests des erreurs de chargement au startup
# ============================================================================

@pytest.mark.unit
def test_startup_with_loading_error(mock_environment):
    """Teste le comportement au startup quand le chargement échoue."""
    with patch("api.main.get_embeddings_model", side_effect=Exception("Load error")), \
         patch("api.main.load_vector_store"), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"):

        from api.main import app

        # L'app devrait gérer l'erreur sans crash
        with pytest.raises(Exception):
            with TestClient(app):
                pass


@pytest.mark.unit
def test_startup_without_mistral_key(mock_environment):
    """Teste le startup sans clé Mistral AI."""
    env_no_key = os.environ.copy()
    del env_no_key["MISTRAL_API_KEY"]  # Supprimer complètement la clé

    with patch.dict(os.environ, env_no_key, clear=True), \
         patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.Mistral") as mock_mistral:

        # Éviter que Mistral soit créé
        mock_mistral.return_value = None

        from api.main import app
        import api.main

        # Forcer mistral_client à None après startup
        api.main.vector_store = Mock()
        api.main.embeddings_model = Mock()
        api.main.mistral_client = None

        with TestClient(app) as client:
            # Force la valeur pour le test
            api.main.mistral_client = None

            # Health check devrait montrer que Mistral n'est pas chargé
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["mistral_client_loaded"] is False


# ============================================================================
# Tests des cas d'erreur dans /search
# ============================================================================

@pytest.mark.unit
def test_search_with_exception(mock_environment):
    """Teste /search quand la recherche lève une exception."""
    mock_vector_store = Mock()
    mock_vector_store.similarity_search_with_score = Mock(
        side_effect=Exception("Search failed")
    )

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()

        with TestClient(app) as client:
            response = client.post("/search", json={"query": "test", "k": 5})
            assert response.status_code == 500


# ============================================================================
# Tests des cas d'erreur dans /ask
# ============================================================================

@pytest.mark.unit
def test_ask_without_mistral_client(mock_environment):
    """Teste /ask quand le client Mistral n'est pas initialisé."""
    mock_vector_store = Mock()

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        # Configurer l'app avec vector_store et embeddings mais sans mistral
        with TestClient(app) as client:
            # Forcer mistral_client à None avant l'appel
            api.main.mistral_client = None

            response = client.post("/ask", json={"question": "test", "k": 5})
            assert response.status_code == 503
            assert "Mistral AI non initialisé" in response.json()["detail"]


@pytest.mark.unit
def test_ask_with_search_exception(mock_environment):
    """Teste /ask quand la recherche vectorielle échoue."""
    mock_vector_store = Mock()
    mock_vector_store.similarity_search_with_score = Mock(
        side_effect=Exception("Search error")
    )
    mock_mistral = Mock()

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral", return_value=mock_mistral), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()
        api.main.mistral_client = mock_mistral

        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "test", "k": 5})
            assert response.status_code == 500


@pytest.mark.unit
def test_ask_with_default_system_prompt(mock_environment):
    """Teste /ask avec le prompt système par défaut."""
    mock_vector_store = Mock()
    mock_vector_store.similarity_search_with_score = Mock(return_value=[
        (Mock(page_content="Test content", metadata={"title": "Test"}), 0.9)
    ])

    mock_mistral = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test answer"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_mistral.chat.complete = Mock(return_value=mock_response)

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral", return_value=mock_mistral), \
         patch("api.main.load_system_prompt", return_value="Default prompt"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()
        api.main.mistral_client = mock_mistral
        api.main.default_system_prompt = "Default prompt"

        with TestClient(app) as client:
            # Test sans system_prompt personnalisé
            response = client.post("/ask", json={"question": "test", "k": 3})
            assert response.status_code == 200


@pytest.mark.unit
def test_ask_with_empty_default_prompt(mock_environment):
    """Teste /ask quand le prompt par défaut est vide (fallback)."""
    mock_vector_store = Mock()
    mock_vector_store.similarity_search_with_score = Mock(return_value=[
        (Mock(page_content="Test", metadata={"title": "Test"}), 0.9)
    ])

    mock_mistral = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Answer"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_mistral.chat.complete = Mock(return_value=mock_response)

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral", return_value=mock_mistral), \
         patch("api.main.load_system_prompt", return_value=""), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()
        api.main.mistral_client = mock_mistral
        api.main.default_system_prompt = None

        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "test"})
            assert response.status_code == 200


# ============================================================================
# Tests du helper load_system_prompt
# ============================================================================

@pytest.mark.unit
def test_load_system_prompt_file_not_found():
    """Teste load_system_prompt avec fichier inexistant."""
    from api.main import load_system_prompt

    with pytest.raises(FileNotFoundError):
        load_system_prompt("/path/that/does/not/exist.md")


@pytest.mark.unit
def test_load_system_prompt_success(tmp_path):
    """Teste load_system_prompt avec un fichier valide."""
    from api.main import load_system_prompt

    # Créer un fichier temporaire
    test_file = tmp_path / "test_prompt.md"
    test_file.write_text("Test prompt content", encoding="utf-8")

    content = load_system_prompt(str(test_file))
    assert content == "Test prompt content"


# ============================================================================
# Tests de /stats avec erreur
# ============================================================================

@pytest.mark.unit
def test_stats_with_exception(mock_environment):
    """Teste /stats quand get_vector_store_stats lève une exception."""
    mock_vector_store = Mock()

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats") as mock_stats:

        # Au startup ça marche, mais après ça échoue
        mock_stats.side_effect = [
            {"num_vectors": 100, "dimension": 1024},  # Pour le startup
            Exception("Stats error")  # Pour l'appel /stats
        ]

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()

        with TestClient(app) as client:
            response = client.get("/stats")
            assert response.status_code == 500


# ============================================================================
# Tests supplémentaires pour /rebuild
# ============================================================================

@pytest.mark.unit
def test_rebuild_concurrent_calls_simple(mock_environment):
    """Teste que deux appels concurrents à /rebuild sont gérés."""
    mock_vector_store = Mock()

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        # Simuler un rebuild en cours
        api.main.rebuild_in_progress = True
        api.main.rebuild_status = {
            "status": "running",
            "message": "En cours",
            "started_at": "2025-01-01T10:00:00",
            "last_update_date": "2025-01-01T09:00:00"
        }

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()

        with TestClient(app) as client:
            response = client.post("/rebuild")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert "déjà en cours" in data["message"]

        # Remettre à False pour ne pas affecter les autres tests
        api.main.rebuild_in_progress = False


@pytest.mark.unit
def test_rebuild_status_various_states(mock_environment):
    """Teste /rebuild/status avec différents états."""
    mock_vector_store = Mock()

    with patch("api.main.get_embeddings_model"), \
         patch("api.main.load_vector_store", return_value=mock_vector_store), \
         patch("api.main.Mistral"), \
         patch("api.main.load_system_prompt", return_value="Test"), \
         patch("api.main.get_vector_store_stats", return_value={"num_vectors": 100, "dimension": 1024}):

        from api.main import app
        import api.main

        api.main.vector_store = mock_vector_store
        api.main.embeddings_model = Mock()

        # Test avec différents états
        test_states = [
            {"status": "idle", "message": "Aucun rebuild"},
            {"status": "running", "message": "En cours"},
            {"status": "success", "message": "Terminé"},
            {"status": "error", "message": "Erreur"},
            {"status": "warning", "message": "Pas de nouveaux événements"}
        ]

        with TestClient(app) as client:
            for state in test_states:
                api.main.rebuild_status = state.copy()
                api.main.rebuild_status["started_at"] = "2025-01-01T10:00:00"
                api.main.rebuild_status["last_update_date"] = None

                response = client.get("/rebuild/status")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == state["status"]
                assert state["message"] in data["message"]
