"""
Tests unitaires pour le module embeddings (embeddings.py).

Ce module teste la génération d'embeddings avec multilingual-e5-large.
"""

import os
from unittest.mock import patch, MagicMock

import pytest
import torch


@pytest.fixture
def mock_environment():
    """Configure l'environnement de test."""
    env_vars = {
        "EMBEDDINGS_MODEL": "intfloat/multilingual-e5-large",
        "EMBEDDINGS_DEVICE": "cpu",
        "EMBEDDINGS_BATCH_SIZE": "16",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.mark.unit
def test_e5embeddings_init_with_cpu(mock_environment):
    """Teste l'initialisation du modèle E5 avec CPU."""
    with patch("embeddings.embeddings.AutoTokenizer") as mock_tokenizer, \
         patch("embeddings.embeddings.AutoModel") as mock_model:

        # Mock des objets
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device="cpu", batch_size=16)

        assert embeddings.device == "cpu"
        assert embeddings.batch_size == 16
        assert embeddings.max_length == 512
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()


@pytest.mark.unit
def test_e5embeddings_device_autodetection(mock_environment):
    """Teste la détection automatique du device."""
    with patch("embeddings.embeddings.AutoTokenizer"), \
         patch("embeddings.embeddings.AutoModel"), \
         patch("embeddings.embeddings.torch.cuda.is_available", return_value=False), \
         patch("embeddings.embeddings.torch.backends.mps.is_available", return_value=False):

        from embeddings.embeddings import E5Embeddings

        # Device None => auto-detect
        embeddings = E5Embeddings(device=None)
        assert embeddings.device == "cpu"


@pytest.mark.unit
def test_e5embeddings_device_cuda_available(mock_environment):
    """Teste la détection de CUDA si disponible."""
    with patch("embeddings.embeddings.AutoTokenizer"), \
         patch("embeddings.embeddings.AutoModel"), \
         patch("embeddings.embeddings.torch.cuda.is_available", return_value=True):

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device=None)
        assert embeddings.device == "cuda"


@pytest.mark.unit
def test_e5embeddings_device_mps_available(mock_environment):
    """Teste la détection de MPS si disponible (et CUDA non disponible)."""
    with patch("embeddings.embeddings.AutoTokenizer"), \
         patch("embeddings.embeddings.AutoModel"), \
         patch("embeddings.embeddings.torch.cuda.is_available", return_value=False), \
         patch("embeddings.embeddings.torch.backends.mps.is_available", return_value=True):

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device=None)
        assert embeddings.device == "mps"


@pytest.mark.unit
def test_e5embeddings_empty_device_string(mock_environment):
    """Teste que les chaînes vides sont traitées comme None."""
    with patch("embeddings.embeddings.AutoTokenizer"), \
         patch("embeddings.embeddings.AutoModel"), \
         patch("embeddings.embeddings.torch.cuda.is_available", return_value=False), \
         patch("embeddings.embeddings.torch.backends.mps.is_available", return_value=False):

        from embeddings.embeddings import E5Embeddings

        # Empty string => auto-detect
        embeddings = E5Embeddings(device="")
        assert embeddings.device == "cpu"


@pytest.mark.unit
def test_average_pool():
    """Teste la fonction average_pool."""
    from embeddings.embeddings import E5Embeddings

    # Créer des tenseurs de test
    # Batch size=2, seq_len=3, hidden_dim=4
    last_hidden_states = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 0.0, 0.0]],  # padding
        [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]]
    ])

    # Attention mask: 1 = token valide, 0 = padding
    attention_mask = torch.tensor([
        [1, 1, 0],  # 2 tokens valides, 1 padding
        [1, 1, 1]   # 3 tokens valides
    ])

    result = E5Embeddings.average_pool(last_hidden_states, attention_mask)

    # Vérifier les dimensions
    assert result.shape == (2, 4)

    # Vérifier les valeurs pour le premier exemple (moyenne de 2 tokens)
    expected_first = torch.tensor([3.0, 4.0, 5.0, 6.0])  # (1+5)/2, (2+6)/2, etc.
    assert torch.allclose(result[0], expected_first)

    # Vérifier les valeurs pour le deuxième exemple (moyenne de 3 tokens)
    expected_second = torch.tensor([6.0, 7.0, 8.0, 9.0])  # (2+6+10)/3, etc.
    assert torch.allclose(result[1], expected_second)


@pytest.mark.unit
def test_embed_documents(mock_environment):
    """Teste l'encodage de documents."""
    with patch("embeddings.embeddings.AutoTokenizer") as mock_tokenizer_class, \
         patch("embeddings.embeddings.AutoModel") as mock_model_class:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        def tokenizer_side_effect(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * batch_size),
                "attention_mask": torch.tensor([[1, 1, 1]] * batch_size)
            }

        mock_tokenizer.side_effect = tokenizer_side_effect

        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        def model_side_effect(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch_size, 3, 1024)
            return mock_output

        mock_model.side_effect = model_side_effect

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device="cpu", batch_size=2)

        # Test encoding
        texts = ["Document 1", "Document 2"]
        result = embeddings.embed_documents(texts)

        # Vérifier le format de sortie
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert len(result[0]) == 1024  # Dimension E5


@pytest.mark.unit
def test_embed_query(mock_environment):
    """Teste l'encodage d'une requête."""
    with patch("embeddings.embeddings.AutoTokenizer") as mock_tokenizer_class, \
         patch("embeddings.embeddings.AutoModel") as mock_model_class:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock model outputs
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 3, 1024)
        mock_model.return_value = mock_output

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device="cpu")

        # Test encoding
        query = "Test query"
        result = embeddings.embed_query(query)

        # Vérifier le format de sortie
        assert isinstance(result, list)
        assert len(result) == 1024  # Dimension E5


@pytest.mark.unit
def test_embed_texts_with_prefix(mock_environment):
    """Teste que les préfixes sont correctement ajoutés."""
    with patch("embeddings.embeddings.AutoTokenizer") as mock_tokenizer_class, \
         patch("embeddings.embeddings.AutoModel") as mock_model_class:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        def tokenizer_side_effect(texts, **kwargs):
            # Vérifier que les préfixes sont présents
            for text in texts:
                assert text.startswith("passage: ") or text.startswith("query: ")
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
                "attention_mask": torch.tensor([[1, 1, 1]] * len(texts))
            }

        mock_tokenizer.side_effect = tokenizer_side_effect

        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 3, 1024)
        mock_model.return_value = mock_output

        from embeddings.embeddings import E5Embeddings

        embeddings = E5Embeddings(device="cpu")

        # Test avec documents (prefix "passage: ")
        embeddings.embed_documents(["Doc 1", "Doc 2"])

        # Test avec query (prefix "query: ")
        embeddings.embed_query("Query")


@pytest.mark.unit
def test_get_embeddings_model_with_env(mock_environment):
    """Teste la factory function avec variables d'environnement."""
    with patch("embeddings.embeddings.E5Embeddings") as mock_e5:
        from embeddings.embeddings import get_embeddings_model

        get_embeddings_model()

        # Vérifier que E5Embeddings est appelé avec le bon model_id
        mock_e5.assert_called_once()
        call_kwargs = mock_e5.call_args[1]
        assert call_kwargs["model_id"] == "intfloat/multilingual-e5-large"
        assert call_kwargs["batch_size"] == 32


@pytest.mark.unit
def test_get_embeddings_model_custom_params(mock_environment):
    """Teste la factory function avec paramètres personnalisés."""
    with patch("embeddings.embeddings.E5Embeddings") as mock_e5:
        from embeddings.embeddings import get_embeddings_model

        get_embeddings_model(
            model_id="custom-model",
            device="cuda",
            batch_size=64
        )

        mock_e5.assert_called_once_with(
            model_id="custom-model",
            device="cuda",
            batch_size=64
        )


@pytest.mark.unit
def test_batch_processing(mock_environment):
    """Teste le traitement par batch."""
    with patch("embeddings.embeddings.AutoTokenizer") as mock_tokenizer_class, \
         patch("embeddings.embeddings.AutoModel") as mock_model_class:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        call_count = 0

        def tokenizer_side_effect(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = len(texts)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * batch_size),
                "attention_mask": torch.tensor([[1, 1, 1]] * batch_size)
            }

        mock_tokenizer.side_effect = tokenizer_side_effect

        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        def model_side_effect(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch_size, 3, 1024)
            return mock_output

        mock_model.side_effect = model_side_effect

        from embeddings.embeddings import E5Embeddings

        # Batch size de 2, mais 5 documents => doit faire 3 appels
        embeddings = E5Embeddings(device="cpu", batch_size=2)
        texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]
        result = embeddings.embed_documents(texts)

        # Vérifier le nombre d'appels (ceil(5/2) = 3)
        assert call_count == 3
        assert len(result) == 5
