"""
Tests unitaires pour le module vectors (vectors.py).

Ce module teste la gestion de la base vectorielle FAISS.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_embeddings():
    """Mock du modèle d'embeddings."""
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    embeddings.embed_query.return_value = [0.15] * 1024
    return embeddings


@pytest.fixture
def mock_documents():
    """Documents de test."""
    return [
        Document(
            page_content="Document 1",
            metadata={"title": "Titre 1", "city": "Paris"}
        ),
        Document(
            page_content="Document 2",
            metadata={"title": "Titre 2", "city": "Lyon"}
        ),
    ]


@pytest.mark.unit
def test_create_vector_store_success(mock_embeddings, mock_documents):
    """Teste la création d'un vector store avec succès."""
    with patch("vectors.vectors.FAISS") as mock_faiss:
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store

        from vectors.vectors import create_vector_store

        result = create_vector_store(mock_documents, mock_embeddings, verbose=True)

        assert result == mock_vector_store
        mock_faiss.from_documents.assert_called_once_with(
            documents=mock_documents,
            embedding=mock_embeddings
        )


@pytest.mark.unit
def test_create_vector_store_empty_documents(mock_embeddings):
    """Teste la création avec une liste de documents vide."""
    from vectors.vectors import create_vector_store

    with pytest.raises(ValueError, match="La liste de documents est vide"):
        create_vector_store([], mock_embeddings)


@pytest.mark.unit
def test_create_vector_store_verbose_logging(mock_embeddings, mock_documents):
    """Teste que le mode verbose affiche les logs."""
    with patch("vectors.vectors.FAISS") as mock_faiss, \
         patch("vectors.vectors.logger") as mock_logger:

        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store

        from vectors.vectors import create_vector_store

        create_vector_store(mock_documents, mock_embeddings, verbose=True)

        # Vérifier que logger.info a été appelé
        assert mock_logger.info.call_count >= 2


@pytest.mark.unit
def test_save_vector_store_success(mock_embeddings, tmp_path):
    """Teste la sauvegarde d'un vector store."""
    mock_vector_store = MagicMock()

    from vectors.vectors import save_vector_store

    save_path = str(tmp_path / "faiss_test")
    save_vector_store(mock_vector_store, save_path, verbose=True)

    # Vérifier que le répertoire a été créé
    assert Path(save_path).exists()

    # Vérifier que save_local a été appelé
    mock_vector_store.save_local.assert_called_once_with(save_path)


@pytest.mark.unit
def test_save_vector_store_creates_directory(mock_embeddings, tmp_path):
    """Teste que save_vector_store crée le répertoire s'il n'existe pas."""
    mock_vector_store = MagicMock()

    from vectors.vectors import save_vector_store

    save_path = str(tmp_path / "nested" / "path" / "faiss")
    assert not Path(save_path).exists()

    save_vector_store(mock_vector_store, save_path)

    # Vérifier que le répertoire a été créé (y compris parents)
    assert Path(save_path).exists()


@pytest.mark.unit
def test_load_vector_store_success(mock_embeddings, tmp_path):
    """Teste le chargement d'un vector store."""
    # Créer un répertoire de test
    load_path = tmp_path / "faiss_test"
    load_path.mkdir()

    with patch("vectors.vectors.FAISS") as mock_faiss:
        mock_vector_store = MagicMock()
        mock_faiss.load_local.return_value = mock_vector_store

        from vectors.vectors import load_vector_store

        result = load_vector_store(str(load_path), mock_embeddings, verbose=True)

        assert result == mock_vector_store
        mock_faiss.load_local.assert_called_once_with(
            str(load_path),
            mock_embeddings,
            allow_dangerous_deserialization=True
        )


@pytest.mark.unit
def test_load_vector_store_not_found(mock_embeddings, tmp_path):
    """Teste le chargement depuis un chemin inexistant."""
    from vectors.vectors import load_vector_store

    non_existent_path = str(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError, match="n'existe pas"):
        load_vector_store(non_existent_path, mock_embeddings)


@pytest.mark.unit
def test_search_similar_documents(mock_embeddings):
    """Teste la recherche de documents similaires."""
    mock_vector_store = MagicMock()

    # Mock des résultats de recherche
    mock_doc1 = Document(
        page_content="Result 1",
        metadata={"title": "Title 1"}
    )
    mock_doc2 = Document(
        page_content="Result 2",
        metadata={"title": "Title 2"}
    )
    mock_vector_store.similarity_search_with_score.return_value = [
        (mock_doc1, 0.85),
        (mock_doc2, 0.75)
    ]

    from vectors.vectors import search_similar_documents

    results = search_similar_documents(
        mock_vector_store,
        "test query",
        k=2,
        verbose=True
    )

    assert len(results) == 2
    assert results[0][0] == mock_doc1
    assert results[0][1] == 0.85
    assert results[1][0] == mock_doc2
    assert results[1][1] == 0.75

    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        "test query",
        k=2
    )


@pytest.mark.unit
def test_search_similar_documents_verbose_logging(mock_embeddings):
    """Teste que le mode verbose affiche les résultats."""
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = [
        (Document(page_content="Test", metadata={"title": "Test"}), 0.9)
    ]

    with patch("vectors.vectors.logger") as mock_logger:
        from vectors.vectors import search_similar_documents

        search_similar_documents(mock_vector_store, "query", k=1, verbose=True)

        # Vérifier que des logs ont été émis
        assert mock_logger.info.call_count >= 2


@pytest.mark.unit
def test_add_documents_to_vector_store(mock_embeddings, mock_documents):
    """Teste l'ajout de documents à un vector store."""
    mock_vector_store = MagicMock()

    from vectors.vectors import add_documents_to_vector_store

    result = add_documents_to_vector_store(
        mock_vector_store,
        mock_documents,
        verbose=True
    )

    assert result == mock_vector_store
    mock_vector_store.add_documents.assert_called_once_with(mock_documents)


@pytest.mark.unit
def test_add_documents_empty_list(mock_embeddings):
    """Teste l'ajout d'une liste vide de documents."""
    mock_vector_store = MagicMock()

    from vectors.vectors import add_documents_to_vector_store

    result = add_documents_to_vector_store(mock_vector_store, [], verbose=False)

    # Ne devrait PAS appeler add_documents pour une liste vide (early return)
    assert result == mock_vector_store
    mock_vector_store.add_documents.assert_not_called()


@pytest.mark.unit
def test_delete_vector_store_success(tmp_path):
    """Teste la suppression d'un vector store."""
    # Créer un répertoire de test avec des fichiers
    vector_store_path = tmp_path / "faiss_test"
    vector_store_path.mkdir()
    (vector_store_path / "index.faiss").touch()
    (vector_store_path / "index.pkl").touch()

    from vectors.vectors import delete_vector_store

    delete_vector_store(str(vector_store_path), verbose=True)

    # Vérifier que le répertoire n'existe plus
    assert not vector_store_path.exists()


@pytest.mark.unit
def test_delete_vector_store_not_found(tmp_path):
    """Teste la suppression d'un vector store inexistant."""
    from vectors.vectors import delete_vector_store

    non_existent_path = str(tmp_path / "does_not_exist")

    # Ne devrait pas lever d'exception
    delete_vector_store(non_existent_path, verbose=False)


@pytest.mark.unit
def test_get_vector_store_stats():
    """Teste la récupération des statistiques du vector store."""
    mock_vector_store = MagicMock()
    mock_vector_store.index.ntotal = 1000

    # Mock des embeddings pour obtenir la dimension
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 1024

    from vectors.vectors import get_vector_store_stats

    stats = get_vector_store_stats(mock_vector_store, verbose=True)

    assert stats["num_vectors"] == 1000
    assert "dimension" in stats


@pytest.mark.unit
def test_get_vector_store_stats_verbose_logging():
    """Teste que les stats sont loggées en mode verbose."""
    mock_vector_store = MagicMock()
    mock_vector_store.index.ntotal = 500

    with patch("vectors.vectors.logger") as mock_logger:
        from vectors.vectors import get_vector_store_stats

        get_vector_store_stats(mock_vector_store, verbose=True)

        # Vérifier que logger a été appelé
        assert mock_logger.info.call_count >= 1
