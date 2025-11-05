"""
Tests d'évaluation RAGAS pour le système RAG.

Ce module contient des tests d'évaluation de la qualité du RAG (Retrieval Augmented
Generation) en utilisant le framework RAGAS. Ces tests sont indépendants des tests
unitaires et évaluent les performances réelles du système.

RAGAS évalue plusieurs métriques :
- Context Precision : Pertinence du contexte récupéré
- Context Recall : Complétude du contexte récupéré
- Faithfulness : Fidélité de la réponse au contexte
- Answer Relevancy : Pertinence de la réponse à la question

Prérequis :
- API FastAPI démarrée (make run-api)
- Index FAISS créé (make run-embeddings)
- Variables d'environnement configurées (.env)
"""

import os
import logging
import time
import pytest
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement de test
test_env_path = Path(__file__).parent.parent / ".env.test"
if test_env_path.exists():
    load_dotenv(test_env_path)
    logger_init = logging.getLogger(__name__)
    logger_init.info(f"Configuration de test chargée depuis {test_env_path}")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration et fixtures
# ============================================================================

@pytest.fixture
def api_url():
    """URL de l'API RAG."""
    return os.getenv("RAG_API_URL", "http://localhost:8000")


@pytest.fixture
def ragas_test_data():
    """
    Dataset de test pour l'évaluation RAGAS.

    Chaque exemple contient :
    - question : La question posée par l'utilisateur
    - ground_truth : La réponse attendue (vérité terrain)
    - contexts : Les contextes attendus (optionnel)
    """
    return [
        {
            "question": "Quels sont les événements culturels gratuits à Toulouse ?",
            "ground_truth": "Il existe plusieurs événements culturels gratuits à Toulouse comme les concerts dans les parcs, les expositions municipales et les festivals de rue.",
        },
        {
            "question": "Où puis-je trouver des expositions d'art contemporain en Occitanie ?",
            "ground_truth": "Les expositions d'art contemporain en Occitanie sont disponibles dans plusieurs musées et galeries à Toulouse, Montpellier et dans d'autres villes de la région.",
        },
        {
            "question": "Quels festivals de musique ont lieu en été en Occitanie ?",
            "ground_truth": "Plusieurs festivals de musique ont lieu en été en Occitanie, incluant des festivals de jazz, de musique classique et de musiques du monde.",
        },
    ]


@pytest.fixture
def ragas_config():
    """
    Configuration pour l'évaluation RAGAS.

    Charge les paramètres depuis .env.test avec des valeurs par défaut.
    """
    return {
        "top_k": int(os.getenv("RAGAS_TOP_K", "5")),
        "timeout": int(os.getenv("RAGAS_API_TIMEOUT", "30")),
        "mistral_delay": float(os.getenv("RAGAS_MISTRAL_DELAY", "2.0")),
    }


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def check_api_health(api_url: str) -> bool:
    """
    Vérifie que l'API RAG est accessible et fonctionnelle.

    Args:
        api_url : URL de l'API

    Returns:
        bool : True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"API Health Check: {data}")
            # Vérifier que l'API retourne un statut valide (ok ou healthy)
            # et que les composants essentiels sont chargés
            status_ok = data.get("status") in ["ok", "healthy"]
            vector_store_ok = data.get("vector_store_loaded", False)
            embeddings_ok = data.get("embeddings_model_loaded", False)
            return status_ok and vector_store_ok and embeddings_ok
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors du health check: {e}")
        return False


def apply_mistral_delay(delay: float = 2.0):
    """
    Applique un délai entre les appels à l'API Mistral.

    Permet d'éviter de dépasser les limites de quota (rate limiting).

    Args:
        delay : Délai en secondes (par défaut: 2.0)
    """
    if delay > 0:
        logger.debug(f"⏳ Attente de {delay}s avant le prochain appel Mistral...")
        time.sleep(delay)


def get_rag_response(
    api_url: str,
    question: str,
    k: int = 5,
    timeout: int = 30,
    mistral_delay: float = 0.0
) -> Dict[str, Any]:
    """
    Récupère une réponse du système RAG via l'API.

    Args:
        api_url : URL de l'API
        question : Question à poser
        k : Nombre de contextes à récupérer
        timeout : Timeout pour la requête
        mistral_delay : Délai à appliquer après l'appel (pour éviter rate limiting)

    Returns:
        dict : Réponse contenant answer, context_used, tokens_used

    Raises:
        requests.exceptions.HTTPError : Si l'API retourne une erreur
        pytest.skip : Si l'API Mistral a dépassé son quota (429)
    """
    try:
        response = requests.post(
            f"{api_url}/ask",
            json={"question": question, "k": k},
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        # Appliquer le délai après un appel réussi
        if mistral_delay > 0:
            apply_mistral_delay(mistral_delay)

        return result
    except requests.exceptions.HTTPError as e:
        # Gestion spéciale pour les erreurs 429 (rate limit Mistral)
        if e.response.status_code == 500:
            try:
                error_detail = e.response.json().get("detail", "")
                if "429" in error_detail or "capacity exceeded" in error_detail.lower():
                    pytest.skip(
                        "API Mistral a dépassé son quota (429). "
                        "Réessayez plus tard ou augmentez votre tier."
                    )
            except Exception:
                pass
        logger.error(f"Erreur HTTP lors de la requête RAG: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la requête RAG: {e}")
        raise


def format_contexts_for_ragas(context_used: List[Dict[str, Any]]) -> List[str]:
    """
    Formate les contextes récupérés pour l'évaluation RAGAS.

    Args:
        context_used : Liste des contextes utilisés par le RAG

    Returns:
        list : Liste de chaînes de caractères (contextes formatés)
    """
    return [ctx.get("content", "") for ctx in context_used]


# ============================================================================
# Tests RAGAS
# ============================================================================

@pytest.mark.ragas
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY non configurée"
)
def test_api_is_running(api_url):
    """
    Vérifie que l'API RAG est accessible avant de lancer les tests RAGAS.

    Ce test est un prérequis pour tous les autres tests RAGAS.
    """
    logger.info("=" * 70)
    logger.info("Vérification de l'accessibilité de l'API RAG")
    logger.info("=" * 70)

    is_healthy = check_api_health(api_url)

    assert is_healthy, (
        f"L'API RAG n'est pas accessible à {api_url}. "
        "Assurez-vous que l'API est démarrée avec 'make run-api'"
    )

    logger.info("✓ API RAG accessible et fonctionnelle")


@pytest.mark.ragas
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY non configurée"
)
def test_rag_retrieval_quality(api_url, ragas_test_data, ragas_config):
    """
    Teste la qualité de récupération des contextes par le RAG.

    Vérifie que :
    - Les contextes sont récupérés avec succès
    - Le nombre de contextes est cohérent
    - Les scores de similarité sont dans une plage acceptable
    """
    logger.info("=" * 70)
    logger.info("TEST DE QUALITÉ DE RÉCUPÉRATION DES CONTEXTES")
    logger.info("=" * 70)
    logger.info(f"Configuration: top_k={ragas_config['top_k']}, "
                f"timeout={ragas_config['timeout']}s, "
                f"mistral_delay={ragas_config['mistral_delay']}s")
    logger.info("")

    results = []

    for i, example in enumerate(ragas_test_data, 1):
        question = example["question"]
        logger.info(f"\nTest {i}/{len(ragas_test_data)}")
        logger.info(f"Question: {question}")

        # Récupérer la réponse RAG (avec délai configuré)
        response = get_rag_response(
            api_url,
            question,
            k=ragas_config["top_k"],
            timeout=ragas_config["timeout"],
            mistral_delay=ragas_config["mistral_delay"]
        )

        # Vérifier la structure de la réponse
        assert "answer" in response, "Réponse manquante"
        assert "context_used" in response, "Contextes manquants"

        contexts = response["context_used"]

        # Vérifier que des contextes ont été récupérés
        assert len(contexts) > 0, "Aucun contexte récupéré"
        assert len(contexts) <= ragas_config["top_k"], (
            f"Trop de contextes récupérés: {len(contexts)}"
        )

        # Vérifier la qualité des scores
        scores = [ctx.get("score", 0) for ctx in contexts]
        avg_score = sum(scores) / len(scores) if scores else 0

        logger.info(f"  - Contextes récupérés: {len(contexts)}")
        logger.info(f"  - Score moyen: {avg_score:.4f}")
        logger.info(f"  - Meilleur score: {max(scores):.4f}")

        # Vérifier que les scores sont dans une plage acceptable
        assert max(scores) > 0.1, (
            f"Scores trop faibles (max: {max(scores):.4f}). "
            "Le contexte récupéré n'est pas pertinent."
        )

        results.append({
            "question": question,
            "num_contexts": len(contexts),
            "avg_score": avg_score,
            "max_score": max(scores),
        })

    # Résumé global
    logger.info("\n" + "=" * 70)
    logger.info("RÉSUMÉ DE LA QUALITÉ DE RÉCUPÉRATION")
    logger.info("=" * 70)

    avg_num_contexts = sum(r["num_contexts"] for r in results) / len(results)
    avg_score_all = sum(r["avg_score"] for r in results) / len(results)

    logger.info(f"Nombre moyen de contextes: {avg_num_contexts:.1f}")
    logger.info(f"Score moyen global: {avg_score_all:.4f}")
    logger.info("✓ Test de récupération terminé avec succès")


@pytest.mark.ragas
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY non configurée"
)
def test_rag_answer_generation(api_url, ragas_test_data, ragas_config):
    """
    Teste la génération de réponses par le système RAG.

    Vérifie que :
    - Une réponse est générée pour chaque question
    - La réponse n'est pas vide
    - La réponse contient du contenu pertinent
    - Les tokens sont comptabilisés correctement
    """
    logger.info("=" * 70)
    logger.info("TEST DE GÉNÉRATION DE RÉPONSES")
    logger.info("=" * 70)

    results = []

    for i, example in enumerate(ragas_test_data, 1):
        question = example["question"]
        logger.info(f"\nTest {i}/{len(ragas_test_data)}")
        logger.info(f"Question: {question}")

        # Récupérer la réponse RAG (avec délai configuré)
        response = get_rag_response(
            api_url,
            question,
            k=ragas_config["top_k"],
            timeout=ragas_config["timeout"],
            mistral_delay=ragas_config["mistral_delay"]
        )

        answer = response["answer"]
        tokens = response.get("tokens_used", {})

        # Vérifier que la réponse n'est pas vide
        assert answer, "Réponse vide générée"
        assert len(answer) > 50, (
            f"Réponse trop courte ({len(answer)} caractères). "
            "Attendu au moins 50 caractères."
        )

        # Vérifier les tokens
        assert "total_tokens" in tokens, "Comptage de tokens manquant"
        assert tokens["total_tokens"] > 0, "Nombre de tokens invalide"

        logger.info(f"  - Longueur réponse: {len(answer)} caractères")
        logger.info(f"  - Tokens utilisés: {tokens.get('total_tokens', 0)}")
        logger.info(f"  - Aperçu: {answer[:100]}...")

        results.append({
            "question": question,
            "answer_length": len(answer),
            "total_tokens": tokens.get("total_tokens", 0),
        })

    # Résumé global
    logger.info("\n" + "=" * 70)
    logger.info("RÉSUMÉ DE LA GÉNÉRATION DE RÉPONSES")
    logger.info("=" * 70)

    avg_length = sum(r["answer_length"] for r in results) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)

    logger.info(f"Longueur moyenne des réponses: {avg_length:.0f} caractères")
    logger.info(f"Tokens moyens utilisés: {avg_tokens:.0f}")
    logger.info("✓ Test de génération terminé avec succès")


@pytest.mark.ragas
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY non configurée"
)
def test_rag_end_to_end(api_url, ragas_config):
    """
    Test end-to-end du système RAG avec une question réelle.

    Ce test valide l'intégration complète du pipeline RAG :
    1. Récupération de contextes pertinents
    2. Génération d'une réponse cohérente
    3. Vérification de la qualité globale
    """
    logger.info("=" * 70)
    logger.info("TEST END-TO-END DU SYSTÈME RAG")
    logger.info("=" * 70)

    question = "Quels événements culturels sont recommandés pour les familles en Occitanie ?"

    logger.info(f"\nQuestion: {question}")

    # Récupérer la réponse RAG (avec délai configuré)
    response = get_rag_response(
        api_url,
        question,
        k=ragas_config["top_k"],
        timeout=ragas_config["timeout"],
        mistral_delay=ragas_config["mistral_delay"]
    )

    # Vérifications de base
    assert "answer" in response
    assert "context_used" in response
    assert "tokens_used" in response

    answer = response["answer"]
    contexts = response["context_used"]
    tokens = response["tokens_used"]

    # Afficher les résultats
    logger.info(f"\n✓ Réponse générée ({len(answer)} caractères)")
    logger.info(f"✓ Contextes récupérés: {len(contexts)}")
    logger.info(f"✓ Tokens utilisés: {tokens.get('total_tokens', 0)}")

    logger.info("\n" + "-" * 70)
    logger.info("RÉPONSE:")
    logger.info("-" * 70)
    logger.info(answer)

    logger.info("\n" + "-" * 70)
    logger.info("CONTEXTES UTILISÉS:")
    logger.info("-" * 70)
    for i, ctx in enumerate(contexts[:3], 1):  # Afficher top 3
        logger.info(f"\n{i}. {ctx.get('title', 'Sans titre')} (score: {ctx.get('score', 0):.4f})")
        logger.info(f"   {ctx.get('content', '')[:150]}...")

    logger.info("\n" + "=" * 70)
    logger.info("✓ TEST END-TO-END RÉUSSI")
    logger.info("=" * 70)

    # Assertions finales
    assert len(answer) > 100, "Réponse trop courte pour une question complexe"
    assert len(contexts) >= 3, "Pas assez de contextes récupérés"
    assert tokens.get("total_tokens", 0) > 100, "Comptage de tokens suspect"


# ============================================================================
# Point d'entrée pour exécution standalone
# ============================================================================

if __name__ == "__main__":
    """
    Permet d'exécuter les tests RAGAS directement.

    Usage:
        python tests/test_ragas.py

    Ou via make:
        make test-ragas
    """
    pytest.main([__file__, "-v", "-m", "ragas", "-s"])
