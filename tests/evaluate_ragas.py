"""
Script d'évaluation RAGAS pour le système RAG.

Ce script évalue la qualité du RAG (Retrieval Augmented Generation) en utilisant
le framework RAGAS. Il génère un rapport avec les métriques suivantes :
- Faithfulness : Fidélité de la réponse au contexte
- Answer Relevancy : Pertinence de la réponse à la question
- Context Precision : Précision du contexte récupéré
- Context Recall : Complétude du contexte récupéré

Prérequis :
- API FastAPI démarrée (make run-api)
- Index FAISS créé (make run-embeddings)
- Variables d'environnement configurées (.env et .env.test)

Usage:
    python tests/evaluate_ragas.py [fichier_questions.json]
    make test-ragas

Arguments:
    fichier_questions.json : Optionnel, chemin vers le fichier de questions
                             (défaut: ragas_test_questions_collected.json)
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================================
# Configuration
# ============================================================================

# Charger les variables d'environnement
env_path = Path(__file__).parent.parent / ".env"
test_env_path = Path(__file__).parent.parent / ".env.test"

if env_path.exists():
    load_dotenv(env_path)
if test_env_path.exists():
    load_dotenv(test_env_path, override=True)

# Configuration
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAGAS_TOP_K = int(os.getenv("RAGAS_TOP_K", "5"))
RAGAS_API_TIMEOUT = float(os.getenv("RAGAS_API_TIMEOUT", "30"))
RAGAS_MISTRAL_DELAY = float(os.getenv("RAGAS_MISTRAL_DELAY", "2.0"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")


# ============================================================================
# Dataset de test
# ============================================================================

def load_test_questions(json_path: str = None) -> List[Dict[str, Any]]:
    """
    Charge les cas de test depuis un fichier JSON.

    Par défaut, utilise UNIQUEMENT le fichier ragas_test_questions_collected.json.
    Si ce fichier n'existe pas, le script s'arrête avec une erreur explicite.

    Pour générer ce fichier: make collect-ragas

    Structure attendue du JSON:
    {
      "test_cases": [
        {
          "id": "test_001",
          "question": "...",
          "answer": "..." ou null,
          "contexts": [...] ou null,
          "ground_truth": "...",
          "category": "...",
          "location": "...",
          "notes": "..."
        }
      ]
    }

    Args:
        json_path : Chemin vers le fichier JSON (optionnel, utilise ragas_test_questions_collected.json si None)

    Returns:
        list : Liste de dictionnaires avec question, answer, contexts, ground_truth
    """
    if json_path is None:
        # Utiliser UNIQUEMENT le fichier collected
        collected_path = Path(__file__).parent / "ragas_test_questions_collected.json"

        if not collected_path.exists():
            print("\n❌ ERREUR: Fichier de données collectées introuvable")
            print(f"   Fichier attendu: {collected_path}")
            print("")
            print("📋 Pour générer ce fichier:")
            print("   1. Démarrez l'API: make run-api")
            print("   2. Collectez les données: make collect-ragas")
            print("   3. Relancez l'évaluation: make test-ragas")
            print("")
            sys.exit(1)

        json_path = collected_path
        print(f"📦 Utilisation du fichier pré-collecté: {collected_path.name}")
    else:
        json_path = Path(json_path)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Nouvelle structure: "test_cases" au lieu de "questions"
        test_cases = data.get("test_cases", [])
        if not test_cases:
            print(f"⚠️  Aucun cas de test trouvé dans {json_path}")
            return []

        print(f"📋 Chargé {len(test_cases)} cas de test depuis {json_path.name}")
        if "description" in data:
            print(f"   Description: {data['description']}")
        if "version" in data:
            print(f"   Version: {data['version']}")

        # Compter combien ont déjà answer/contexts pré-collectés
        pre_collected = sum(1 for tc in test_cases if tc.get("answer") and tc.get("contexts"))
        if pre_collected > 0:
            print(f"   ✓ {pre_collected}/{len(test_cases)} cas avec answer/contexts pré-collectés")
        else:
            print("   ⚠️  Aucun cas pré-collecté. Les réponses seront collectées dynamiquement.")

        return test_cases

    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {json_path}")
        print("   Utilisation des cas de test par défaut")
        # Cas de test de fallback
        return [
            {
                "id": "fallback_001",
                "question": "Quels sont les événements culturels gratuits à Toulouse ?",
                "answer": None,
                "contexts": None,
                "ground_truth": "Il existe plusieurs événements culturels gratuits à Toulouse comme les concerts dans les parcs, les expositions municipales et les festivals de rue.",
            },
            {
                "id": "fallback_002",
                "question": "Où puis-je trouver des expositions d'art contemporain en Occitanie ?",
                "answer": None,
                "contexts": None,
                "ground_truth": "Les expositions d'art contemporain en Occitanie sont disponibles dans plusieurs musées et galeries à Toulouse, Montpellier et dans d'autres villes de la région.",
            },
            {
                "id": "fallback_003",
                "question": "Quels festivals de musique ont lieu en été en Occitanie ?",
                "answer": None,
                "contexts": None,
                "ground_truth": "Plusieurs festivals de musique ont lieu en été en Occitanie, incluant des festivals de jazz, de musique classique et de musiques du monde.",
            },
        ]
    except json.JSONDecodeError as e:
        print(f"❌ Erreur lors du parsing JSON: {e}")
        print("   Vérifiez la syntaxe du fichier JSON")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des cas de test: {e}")
        sys.exit(1)


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def check_api_health() -> bool:
    """
    Vérifie que l'API RAG est accessible et fonctionnelle.

    Returns:
        bool : True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status_ok = data.get("status") in ["ok", "healthy"]
            vector_store_ok = data.get("vector_store_loaded", False)
            embeddings_ok = data.get("embeddings_model_loaded", False)
            return status_ok and vector_store_ok and embeddings_ok
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors du health check: {e}")
        return False


def get_rag_response(question: str) -> Dict[str, Any]:
    """
    Récupère une réponse du système RAG via l'API.

    Args:
        question : Question à poser

    Returns:
        dict : Réponse contenant answer, context_used, tokens_used

    Raises:
        requests.exceptions.HTTPError : Si l'API retourne une erreur
    """
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "k": RAGAS_TOP_K},
            timeout=RAGAS_API_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()

        # Appliquer le délai après un appel réussi
        if RAGAS_MISTRAL_DELAY > 0:
            time.sleep(RAGAS_MISTRAL_DELAY)

        return result
    except requests.exceptions.HTTPError as e:
        # Gestion spéciale pour les erreurs 429 (rate limit Mistral)
        if e.response.status_code == 500:
            try:
                error_detail = e.response.json().get("detail", "")
                if "429" in error_detail or "capacity exceeded" in error_detail.lower():
                    print(f"\n⚠️  API Mistral a dépassé son quota (429) pour la question: {question}")
                    print("   Réessayez plus tard ou augmentez votre tier.")
                    return None
            except Exception:
                pass
        print(f"❌ Erreur HTTP lors de la requête RAG: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de la requête RAG: {e}")
        raise


def format_contexts_for_ragas(context_used: List[Dict[str, Any]]) -> List[str]:
    """
    Formate les contextes récupérés pour l'évaluation RAGAS.

    Utilise le contenu de l'événement pour l'évaluation RAGAS.
    Cela permet à RAGAS d'évaluer la pertinence basée sur le contenu
    textuel complet des événements récupérés.

    Args:
        context_used : Liste des contextes utilisés par le RAG

    Returns:
        list : Liste de contenus d'événements (strings)
    """
    return [ctx.get("content", "") for ctx in context_used]


def collect_rag_data(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collecte les données du système RAG pour tous les cas de test.

    Si un cas de test a déjà answer et contexts (pré-collectés), ils sont utilisés.
    Sinon, le script interroge l'API RAG pour collecter ces données dynamiquement.

    Args:
        test_cases : Liste des cas de test chargés depuis JSON

    Returns:
        list : Liste de dictionnaires avec question, answer, contexts, ground_truth
    """
    print("\n" + "=" * 70)
    print("📊 COLLECTE DES DONNÉES POUR L'ÉVALUATION RAGAS")
    print("=" * 70)
    print(f"\nNombre de cas de test: {len(test_cases)}")
    print(f"Configuration: top_k={RAGAS_TOP_K}, timeout={RAGAS_API_TIMEOUT}s, delay={RAGAS_MISTRAL_DELAY}s")
    print("")

    results = []
    collected_count = 0
    pre_collected_count = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        test_id = test_case.get("id", f"test_{i:03d}")

        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"   Question: {question}")

        # Vérifier si answer et contexts sont déjà fournis
        if test_case.get("answer") and test_case.get("contexts"):
            print("   ✓ Données pré-collectées trouvées dans le JSON")
            answer = test_case["answer"]
            contexts = test_case["contexts"]

            # Vérifier la validité des données pré-collectées
            if not answer or len(answer) < 50:
                print(f"   ⚠️  Réponse pré-collectée trop courte ({len(answer)} caractères), collecte dynamique...")
            elif not contexts or len(contexts) == 0:
                print("   ⚠️  Aucun contexte pré-collecté, collecte dynamique...")
            else:
                # Utiliser les données pré-collectées
                # contexts peut être une liste de strings ou une liste de dicts
                if isinstance(contexts[0], dict):
                    # Format API: list of dicts with 'content' key
                    formatted_contexts = format_contexts_for_ragas(contexts)
                else:
                    # Format RAGAS: list of strings (déjà formaté)
                    formatted_contexts = contexts

                results.append({
                    "question": question,
                    "answer": answer,
                    "contexts": formatted_contexts,
                    "ground_truth": ground_truth
                })

                print(f"   ✅ Pré-collecté: {len(answer)} caractères, {len(formatted_contexts)} contextes")
                pre_collected_count += 1
                continue

        # Pas de données pré-collectées ou invalides -> collecte dynamique
        print("   🔍 Collecte dynamique via API RAG...")

        # Récupérer la réponse RAG
        response = get_rag_response(question)

        if response is None:
            print("   ⚠️  Cas ignoré (erreur 429 ou timeout)")
            continue

        # Vérifier la structure de la réponse
        if "answer" not in response or "context_used" not in response:
            print("   ❌ Réponse invalide (structure incorrecte)")
            continue

        answer = response["answer"]
        contexts = response["context_used"]

        # Vérifier que la réponse et les contextes sont valides
        if not answer or len(answer) < 50:
            print(f"   ⚠️  Réponse trop courte ({len(answer)} caractères), ignorée")
            continue

        if len(contexts) == 0:
            print("   ⚠️  Aucun contexte récupéré, cas ignoré")
            continue

        # Collecter les données
        results.append({
            "question": question,
            "answer": answer,
            "contexts": format_contexts_for_ragas(contexts),
            "ground_truth": ground_truth
        })

        print(f"   ✅ Collecté: {len(answer)} caractères, {len(contexts)} contextes")
        collected_count += 1

    print(f"\n{'=' * 70}")
    print(f"✅ Collecte terminée: {len(results)}/{len(test_cases)} cas traités")
    if pre_collected_count > 0:
        print(f"   - {pre_collected_count} cas pré-collectés")
    if collected_count > 0:
        print(f"   - {collected_count} cas collectés dynamiquement")
    print("=" * 70)

    return results


def generate_ragas_report(ragas_data: List[Dict[str, Any]]):
    """
    Génère le rapport final RAGAS avec les métriques calculées.

    Args:
        ragas_data : Liste de dictionnaires avec question, answer, contexts, ground_truth
    """
    if not ragas_data:
        print("\n⚠️  Aucune donnée à évaluer. Toutes les questions ont été ignorées.")
        print("   Causes possibles:")
        print("   - API Mistral a dépassé son quota (429)")
        print("   - Réponses trop courtes ou contextes manquants")
        print("   - Problème de connexion à l'API")
        return

    print("\n" + "=" * 70)
    print("🎯 GÉNÉRATION DU RAPPORT RAGAS")
    print("=" * 70)

    try:
        # Créer le dataset pour RAGAS
        dataset_dict = {
            "question": [r["question"] for r in ragas_data],
            "answer": [r["answer"] for r in ragas_data],
            "contexts": [r["contexts"] for r in ragas_data],
            "ground_truth": [r["ground_truth"] for r in ragas_data],
        }

        dataset = Dataset.from_dict(dataset_dict)

        print(f"\n📊 Évaluation de {len(ragas_data)} questions...")
        print("⏳ Calcul des métriques RAGAS en cours...")
        print("   (Cette opération peut prendre 30-60 secondes)")
        print("")

        # Vérifier la clé API
        if not MISTRAL_API_KEY:
            print("❌ MISTRAL_API_KEY non configurée. Impossible d'évaluer avec RAGAS.")
            return

        # Configurer le LLM Mistral AI pour RAGAS
        print(f"⚙️  Configuration LLM: model={MISTRAL_MODEL}, timeout={RAGAS_API_TIMEOUT}s, delay={RAGAS_MISTRAL_DELAY}s")
        print("   Note: RAGAS effectue plusieurs appels API pour calculer les métriques")
        print("   En cas d'erreur 429, augmentez RAGAS_MISTRAL_DELAY dans .env.test")
        print("")

        llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            api_key=MISTRAL_API_KEY,
            temperature=0.0,
            max_retries=3,
            timeout=RAGAS_API_TIMEOUT
        )

        # Configurer les embeddings pour RAGAS (utilise E5 au lieu d'OpenAI)
        print(f"📦 Configuration embeddings: {EMBEDDINGS_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Évaluer avec les métriques RAGAS
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Délai avant l'évaluation pour récupération du quota
        if RAGAS_MISTRAL_DELAY > 0:
            print(f"⏳ Attente de {RAGAS_MISTRAL_DELAY}s avant d'évaluer (récupération du quota API)...")
            time.sleep(RAGAS_MISTRAL_DELAY)

        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

        # Convertir en DataFrame si nécessaire
        df = None
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()

        # Afficher les scores détaillés par question
        if df is not None and len(df) > 0:
            print("\n" + "=" * 70)
            print("📊 SCORES DÉTAILLÉS PAR QUESTION")
            print("=" * 70)
            print("")

            for idx, row in df.iterrows():
                question_text = ragas_data[idx]["question"]
                # Tronquer la question si trop longue
                if len(question_text) > 60:
                    question_text = question_text[:57] + "..."

                print(f"Question {idx + 1}: {question_text}")
                print("-" * 70)

                for metric in metrics:
                    metric_name = metric.name
                    if metric_name in df.columns:
                        score = row[metric_name]
                        # Emoji selon le score
                        if score >= 0.8:
                            emoji = "✅"
                        elif score >= 0.6:
                            emoji = "⚠️ "
                        else:
                            emoji = "❌"
                        print(f"  {emoji} {metric_name:25s}: {score:.4f}")
                print("")

        # Afficher le rapport des moyennes
        print("=" * 70)
        print("📈 MÉTRIQUES RAGAS (MOYENNES)")
        print("=" * 70)
        print("")

        # Extraire les scores moyens
        scores = {}

        if df is not None:
            # Calculer les moyennes pour chaque métrique
            for metric in metrics:
                metric_name = metric.name
                if metric_name in df.columns:
                    score = df[metric_name].mean()
                    scores[metric_name] = score
                    # Afficher avec un emoji selon le score
                    if score >= 0.8:
                        emoji = "✅"
                    elif score >= 0.6:
                        emoji = "⚠️ "
                    else:
                        emoji = "❌"
                    print(f"  {emoji} {metric_name:25s}: {score:.4f}")
        else:
            # Fallback : essayer d'accéder directement aux attributs
            for metric in metrics:
                metric_name = metric.name
                if hasattr(result, metric_name):
                    score = getattr(result, metric_name)
                    if isinstance(score, (int, float)):
                        scores[metric_name] = score
                        # Afficher avec un emoji selon le score
                        if score >= 0.8:
                            emoji = "✅"
                        elif score >= 0.6:
                            emoji = "⚠️ "
                        else:
                            emoji = "❌"
                        print(f"  {emoji} {metric_name:25s}: {score:.4f}")

        print("")
        print("=" * 70)
        print("INTERPRÉTATION DES SCORES")
        print("=" * 70)
        print("")
        print("  ✅ Faithfulness (Fidélité) [0-1]:")
        print("     Mesure si la réponse est fidèle au contexte récupéré")
        print("     > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer")
        print("")
        print("  ✅ Answer Relevancy (Pertinence) [0-1]:")
        print("     Mesure la pertinence de la réponse à la question")
        print("     > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer")
        print("")
        print("  ✅ Context Precision (Précision du contexte) [0-1]:")
        print("     Mesure la précision du contexte récupéré")
        print("     > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer")
        print("")
        print("  ✅ Context Recall (Rappel du contexte) [0-1]:")
        print("     Mesure la complétude du contexte récupéré")
        print("     > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer")
        print("")
        print("=" * 70)
        print("RECOMMANDATIONS")
        print("=" * 70)
        print("")

        # Analyser les résultats et donner des recommandations
        if scores.get("faithfulness", 0) < 0.7:
            print("  ⚠️  Faithfulness faible :")
            print("     - Vérifiez que les réponses restent fidèles au contexte")
            print("     - Ajustez le prompt système pour éviter les hallucinations")
            print("")

        if scores.get("answer_relevancy", 0) < 0.7:
            print("  ⚠️  Answer Relevancy faible :")
            print("     - Vérifiez que les réponses adressent bien la question")
            print("     - Améliorez la qualité du prompt d'enrichissement")
            print("")

        if scores.get("context_precision", 0) < 0.7:
            print("  ⚠️  Context Precision faible :")
            print("     - Améliorez la qualité des embeddings (modèle, chunking)")
            print("     - Ajustez les paramètres de recherche (top_k, seuil)")
            print("")

        if scores.get("context_recall", 0) < 0.7:
            print("  ⚠️  Context Recall faible :")
            print("     - Augmentez le nombre de contextes récupérés (top_k)")
            print("     - Vérifiez la complétude de votre base de données")
            print("")

        # Si tous les scores sont bons
        if all(s >= 0.7 for s in scores.values()):
            print("  ✅ Tous les scores sont satisfaisants !")
            print("     Votre système RAG fonctionne correctement.")
            print("")

        print("=" * 70)
        print("✅ RAPPORT RAGAS TERMINÉ")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Erreur lors de la génération du rapport RAGAS: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Point d'entrée principal
# ============================================================================

def main():
    """
    Point d'entrée principal du script d'évaluation RAGAS.
    """
    # Parser les arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS du système RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Utiliser le fichier par défaut (ragas_test_questions_collected.json)
  python tests/evaluate_ragas.py

  # Utiliser un fichier spécifique
  python tests/evaluate_ragas.py tests/ragas_test_questions_collected.json
  python tests/evaluate_ragas.py tests/my_custom_questions.json
        """
    )
    parser.add_argument(
        "questions_file",
        nargs="?",
        default=None,
        help="Chemin vers le fichier de questions JSON (défaut: ragas_test_questions_collected.json)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎯 ÉVALUATION RAGAS DU SYSTÈME RAG")
    print("=" * 70)
    print("\nPrérequis:")
    print("  - API RAG démarrée: make run-api")
    print("  - Index FAISS créé: make run-embeddings")
    print("  - MISTRAL_API_KEY configurée dans .env")
    print("")

    # Vérifier que l'API est accessible
    print("🔍 Vérification de l'API RAG...")
    if not check_api_health():
        print("\n❌ L'API RAG n'est pas accessible ou non fonctionnelle")
        print(f"   URL: {API_URL}")
        print("   Assurez-vous que l'API est démarrée avec 'make run-api'")
        sys.exit(1)

    print("✅ API RAG accessible et fonctionnelle\n")

    # Charger les questions de test
    test_questions = load_test_questions(args.questions_file)
    if not test_questions:
        print("\n❌ Aucune question de test disponible. Vérifiez le fichier JSON.")
        sys.exit(1)

    # Collecter les données
    ragas_data = collect_rag_data(test_questions)

    # Générer le rapport
    if ragas_data:
        generate_ragas_report(ragas_data)
    else:
        print("\n❌ Aucune donnée collectée. Impossible de générer le rapport.")
        sys.exit(1)


if __name__ == "__main__":
    main()
