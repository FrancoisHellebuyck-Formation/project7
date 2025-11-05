"""
Tests spécifiques pour les fonctionnalités de rebuild de l'API.

Ces tests couvrent la fonction run_rebuild_pipeline et les endpoints associés
pour atteindre l'objectif de 80% de couverture.
"""

import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

import pytest


@pytest.fixture
def mock_environment():
    """Configure les variables d'environnement."""
    env_vars = {
        "FAISS_INDEX_PATH": "data/faiss_index",
        "MONGODB_URI": "mongodb://localhost:27017/",
        "MONGODB_DB_NAME": "OA_TEST",
        "MONGODB_COLLECTION_NAME_EVENTS": "events",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


# ============================================================================
# Tests de la fonction run_rebuild_pipeline
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_no_last_execution(mock_environment):
    """Teste run_rebuild_pipeline quand il n'y a pas de dernière exécution."""
    with patch("pymongo.MongoClient") as mock_mongo_class:
        # Mock MongoDB - aucune exécution précédente
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None  # Pas d'exécution

        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_client.close = Mock()
        mock_mongo_class.return_value = mock_client

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset le statut
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {
            "status": "idle",
            "message": "Test",
            "started_at": None,
            "last_update_date": None
        }

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier que le statut n'est pas "warning" (pas de vérification d'événements)
        assert api.main.rebuild_status["status"] in ["idle", "error"]
        assert not api.main.rebuild_in_progress


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_with_no_new_events(mock_environment):
    """Teste run_rebuild_pipeline quand il n'y a pas de nouveaux événements."""
    with patch("pymongo.MongoClient") as mock_mongo_class:
        # Mock MongoDB
        mock_client = MagicMock()
        mock_db = MagicMock()

        # Mock de last_update collection
        mock_last_update_collection = MagicMock()
        mock_last_update_collection.find_one.return_value = {
            "pipeline_run_date": datetime(2025, 1, 1, 10, 0, 0)
        }

        # Mock de events collection
        mock_events_collection = MagicMock()
        mock_events_collection.count_documents.return_value = 0  # Aucun nouvel événement

        # Configurer __getitem__ pour retourner la bonne collection
        def get_collection(name):
            if name == "last_update":
                return mock_last_update_collection
            elif name == "events":
                return mock_events_collection
            return MagicMock()

        mock_db.__getitem__.side_effect = get_collection
        mock_client.__getitem__.return_value = mock_db
        mock_client.close = Mock()
        mock_mongo_class.return_value = mock_client

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {"status": "idle", "message": "", "started_at": None, "last_update_date": None}

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier
        assert api.main.rebuild_status["status"] == "warning"
        assert "Pas de nouveaux événements" in api.main.rebuild_status["message"]
        assert not api.main.rebuild_in_progress


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_success(mock_environment):
    """Teste run_rebuild_pipeline avec succès complet."""
    with patch("pymongo.MongoClient") as mock_mongo_class, \
         patch("asyncio.create_subprocess_exec") as mock_subprocess, \
         patch("api.main.load_vector_store") as mock_load_vs, \
         patch("api.main.get_vector_store_stats") as mock_stats:

        # Mock MongoDB
        mock_client = MagicMock()
        mock_db = MagicMock()

        mock_last_update = MagicMock()
        mock_last_update.find_one.return_value = {
            "pipeline_run_date": datetime(2025, 1, 1, 10, 0, 0)
        }

        mock_events = MagicMock()
        mock_events.count_documents.return_value = 10  # Il y a de nouveaux événements

        def get_collection(name):
            if name == "last_update":
                return mock_last_update
            return mock_events

        mock_db.__getitem__.side_effect = get_collection
        mock_client.__getitem__.return_value = mock_db
        mock_client.close = Mock()
        mock_mongo_class.return_value = mock_client

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"Success", b""))
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Mock vector store reload
        mock_vector_store = Mock()
        mock_load_vs.return_value = mock_vector_store
        mock_stats.return_value = {"num_vectors": 1000, "dimension": 1024}

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {"status": "idle", "message": "", "started_at": None, "last_update_date": None}
        api.main.embeddings_model = Mock()
        api.main.FAISS_INDEX_PATH = "test/path"

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier
        assert api.main.rebuild_status["status"] == "success"
        assert "succès" in api.main.rebuild_status["message"].lower()
        assert not api.main.rebuild_in_progress
        # Vérifier que le vector store a été rechargé
        mock_load_vs.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_subprocess_failure(mock_environment):
    """Teste run_rebuild_pipeline quand le subprocess échoue."""
    with patch("pymongo.MongoClient") as mock_mongo_class, \
         patch("asyncio.create_subprocess_exec") as mock_subprocess:

        # Mock MongoDB
        mock_client = MagicMock()
        mock_db = MagicMock()

        mock_last_update = MagicMock()
        mock_last_update.find_one.return_value = {
            "pipeline_run_date": datetime(2025, 1, 1, 10, 0, 0)
        }

        mock_events = MagicMock()
        mock_events.count_documents.return_value = 10

        def get_collection(name):
            if name == "last_update":
                return mock_last_update
            return mock_events

        mock_db.__getitem__.side_effect = get_collection
        mock_client.__getitem__.return_value = mock_db
        mock_client.close = Mock()
        mock_mongo_class.return_value = mock_client

        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error occurred"))
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {"status": "idle", "message": "", "started_at": None, "last_update_date": None}

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier
        assert api.main.rebuild_status["status"] == "error"
        assert "Échec" in api.main.rebuild_status["message"]
        assert not api.main.rebuild_in_progress


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_reload_failure(mock_environment):
    """Teste quand le rechargement du vector store échoue."""
    with patch("pymongo.MongoClient") as mock_mongo_class, \
         patch("asyncio.create_subprocess_exec") as mock_subprocess, \
         patch("api.main.load_vector_store") as mock_load_vs:

        # Mock MongoDB
        mock_client = MagicMock()
        mock_db = MagicMock()

        mock_last_update = MagicMock()
        mock_last_update.find_one.return_value = {
            "pipeline_run_date": datetime(2025, 1, 1, 10, 0, 0)
        }

        mock_events = MagicMock()
        mock_events.count_documents.return_value = 10

        def get_collection(name):
            if name == "last_update":
                return mock_last_update
            return mock_events

        mock_db.__getitem__.side_effect = get_collection
        mock_client.__getitem__.return_value = mock_db
        mock_client.close = Mock()
        mock_mongo_class.return_value = mock_client

        # Mock subprocess success
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"Success", b""))
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Mock vector store reload failure
        mock_load_vs.side_effect = Exception("Failed to load")

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {"status": "idle", "message": "", "started_at": None, "last_update_date": None}
        api.main.embeddings_model = Mock()

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier
        assert api.main.rebuild_status["status"] == "success_with_warning"
        assert "échec du rechargement" in api.main.rebuild_status["message"].lower()
        assert not api.main.rebuild_in_progress


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rebuild_pipeline_exception(mock_environment):
    """Teste quand une exception est levée pendant le rebuild."""
    with patch("pymongo.MongoClient") as mock_mongo_class:
        # Mock MongoDB qui lève une exception
        mock_mongo_class.side_effect = Exception("MongoDB connection failed")

        # Import et test
        from api.main import run_rebuild_pipeline
        import api.main

        # Reset
        api.main.rebuild_in_progress = False
        api.main.rebuild_status = {"status": "idle", "message": "", "started_at": None, "last_update_date": None}

        # Exécuter
        await run_rebuild_pipeline()

        # Vérifier
        assert api.main.rebuild_status["status"] == "error"
        assert "Erreur lors du rebuild" in api.main.rebuild_status["message"]
        assert not api.main.rebuild_in_progress
