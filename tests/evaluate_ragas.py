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
                             (défaut: ragas_data/ragas_test_questions_collected.json)
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
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

    Par défaut, utilise UNIQUEMENT le fichier ragas_data/ragas_test_questions_collected.json.
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
        json_path : Chemin vers le fichier JSON (optionnel, utilise ragas_data/ragas_test_questions_collected.json si None)

    Returns:
        list : Liste de dictionnaires avec question, answer, contexts, ground_truth
    """
    if json_path is None:
        # Utiliser UNIQUEMENT le fichier collected
        collected_path = (
            Path(__file__).parent / "ragas_data" / "ragas_test_questions_collected.json"
        )

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


def validate_ragas_data(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Valide que les données RAGAS sont complètes dans le fichier JSON.

    Cette fonction ne fait AUCUN appel API et ne lance PAS le RAG.
    Elle vérifie uniquement que chaque cas de test contient toutes les données:
    - question
    - answer (pré-collecté via make collect-ragas)
    - contexts (pré-collectés via make collect-ragas)
    - ground_truth

    Si des données sont manquantes, le script s'arrête et affiche un message
    clair indiquant comment collecter les données manquantes.

    Args:
        test_cases : Liste des cas de test chargés depuis JSON

    Returns:
        list : Liste de dictionnaires valides avec question, answer, contexts, ground_truth

    Raises:
        SystemExit : Si des données manquantes ou invalides sont détectées
    """
    print("\n" + "=" * 70)
    print("✅ VALIDATION DES DONNÉES RAGAS")
    print("=" * 70)
    print(f"Nombre de cas de test à valider: {len(test_cases)}\n")

    results = []
    valid_count = 0
    invalid_cases = []

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case.get("id", f"test_{i:03d}")
        question = test_case.get("question")
        answer = test_case.get("answer")
        contexts = test_case.get("contexts")
        ground_truth = test_case.get("ground_truth")

        print(f"[{i}/{len(test_cases)}] {test_id}")

        # Vérifier que toutes les données requises sont présentes
        errors = []

        if not question:
            errors.append("question manquante")
        if not answer:
            errors.append("answer manquante")
        elif len(answer) < 50:
            errors.append(f"answer trop courte ({len(answer)} caractères)")
        if not contexts or len(contexts) == 0:
            errors.append("contexts manquants")
        if not ground_truth:
            errors.append("ground_truth manquant")

        if errors:
            print(f"   ❌ INVALIDE: {', '.join(errors)}")
            invalid_cases.append({
                "id": test_id,
                "errors": errors
            })
            continue

        # Formater les contextes si nécessaire
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

        print(f"   ✅ VALIDE: {len(answer)} caractères, {len(formatted_contexts)} contextes")
        valid_count += 1

    print(f"\n{'=' * 70}")
    print("📊 RÉSULTAT DE LA VALIDATION")
    print("=" * 70)
    print(f"✅ Cas valides: {valid_count}/{len(test_cases)}")
    print(f"❌ Cas invalides: {len(invalid_cases)}/{len(test_cases)}")

    if invalid_cases:
        print("\n⚠️  CAS INVALIDES DÉTECTÉS:")
        for case in invalid_cases:
            print(f"   - {case['id']}: {', '.join(case['errors'])}")
        print("\n💡 Pour collecter les données manquantes:")
        print("   1. Assurez-vous que l'API RAG est lancée: make run-api")
        print("   2. Lancez la collecte des données: make collect-ragas")
        print("   3. Relancez l'évaluation: make test-ragas")
        print("=" * 70)
        sys.exit(1)

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

        # Générer le rapport HTML
        try:
            generate_html_report(ragas_data, df, scores, metrics)
        except Exception as html_error:
            print(f"\n⚠️  Erreur lors de la génération du rapport HTML: {html_error}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Erreur lors de la génération du rapport RAGAS: {e}")
        import traceback
        traceback.print_exc()


def generate_html_report(
    ragas_data: List[Dict[str, Any]],
    df,
    scores: Dict[str, float],
    metrics: list,
    output_path: str = "rapport/ragas/ragas_report.html"
) -> None:
    """
    Génère un rapport HTML des résultats d'évaluation RAGAS.

    Args:
        ragas_data: Liste des cas de test
        df: DataFrame avec les résultats détaillés
        scores: Dictionnaire des scores moyens
        metrics: Liste des métriques évaluées
        output_path: Chemin du fichier HTML de sortie
    """
    # Créer le répertoire de sortie si nécessaire
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Générer le timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Template HTML
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation RAGAS</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .timestamp {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}

        .metric-name {{
            font-size: 1em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .metric-excellent {{
            color: #28a745;
        }}

        .metric-good {{
            color: #ffc107;
        }}

        .metric-poor {{
            color: #dc3545;
        }}

        .metric-bar {{
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .metric-bar-fill {{
            height: 100%;
            transition: width 0.5s ease;
        }}

        .bar-excellent {{
            background: linear-gradient(90deg, #28a745, #20c997);
        }}

        .bar-good {{
            background: linear-gradient(90deg, #ffc107, #fd7e14);
        }}

        .bar-poor {{
            background: linear-gradient(90deg, #dc3545, #c82333);
        }}

        .questions-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .questions-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        .questions-table td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}

        .questions-table tr:hover {{
            background: #f8f9fa;
        }}

        .question-text {{
            font-weight: 500;
            color: #333;
            margin-bottom: 5px;
        }}

        .score-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-excellent {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-good {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge-poor {{
            background: #f8d7da;
            color: #721c24;
        }}

        .interpretation {{
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .interpretation h3 {{
            color: #2196F3;
            margin-bottom: 15px;
        }}

        .interpretation ul {{
            list-style: none;
            padding-left: 0;
        }}

        .interpretation li {{
            padding: 10px 0;
            border-bottom: 1px solid #cce5ff;
        }}

        .interpretation li:last-child {{
            border-bottom: none;
        }}

        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .recommendations h3 {{
            color: #856404;
            margin-bottom: 15px;
        }}

        .recommendations ul {{
            padding-left: 20px;
        }}

        .recommendations li {{
            margin-bottom: 10px;
        }}

        .success-message {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: 500;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
            }}

            .metric-card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Rapport d'Évaluation RAGAS</h1>
            <div class="subtitle">Système RAG - Événements Culturels Occitanie</div>
        </div>

        <div class="timestamp">
            Généré le {timestamp}
        </div>

        <div class="content">
            <!-- Métriques moyennes -->
            <div class="section">
                <h2 class="section-title">📈 Métriques Moyennes</h2>
                <div class="metrics-grid">
"""

    # Ajouter les cartes de métriques
    for metric in metrics:
        metric_name = metric.name
        if metric_name in scores:
            score = scores[metric_name]
            score_percent = int(score * 100)

            # Déterminer la classe CSS selon le score
            if score >= 0.8:
                value_class = "metric-excellent"
                bar_class = "bar-excellent"
                badge_class = "badge-excellent"
                badge_text = "Excellent"
            elif score >= 0.6:
                value_class = "metric-good"
                bar_class = "bar-good"
                badge_class = "badge-good"
                badge_text = "Bon"
            else:
                value_class = "metric-poor"
                bar_class = "bar-poor"
                badge_class = "badge-poor"
                badge_text = "À améliorer"

            # Nom de métrique formaté
            metric_display = metric_name.replace("_", " ").title()

            html_content += f"""
                    <div class="metric-card">
                        <div class="metric-name">{metric_display}</div>
                        <div class="metric-value {value_class}">{score:.3f}</div>
                        <span class="score-badge {badge_class}">{badge_text}</span>
                        <div class="metric-bar">
                            <div class="metric-bar-fill {bar_class}" style="width: {score_percent}%"></div>
                        </div>
                    </div>
"""

    html_content += """
                </div>
            </div>

            <!-- Scores détaillés par question -->
            <div class="section">
                <h2 class="section-title">📋 Scores Détaillés par Question</h2>
                <table class="questions-table">
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th>Question</th>
"""

    # En-têtes des métriques
    for metric in metrics:
        metric_display = metric.name.replace("_", " ").title()
        html_content += f"                            <th style=\"width: 120px; text-align: center;\">{metric_display}</th>\n"

    html_content += """
                        </tr>
                    </thead>
                    <tbody>
"""

    # Lignes de résultats
    if df is not None and len(df) > 0:
        for idx, row in df.iterrows():
            question_text = ragas_data[idx]["question"]
            # Tronquer si nécessaire
            if len(question_text) > 100:
                question_text = question_text[:97] + "..."

            html_content += f"""
                        <tr>
                            <td style="text-align: center; font-weight: bold;">{idx + 1}</td>
                            <td><div class="question-text">{question_text}</div></td>
"""

            # Scores pour chaque métrique
            for metric in metrics:
                metric_name = metric.name
                if metric_name in df.columns:
                    score = row[metric_name]
                    score_str = f"{score:.3f}"

                    # Badge selon le score
                    if score >= 0.8:
                        badge_class = "badge-excellent"
                    elif score >= 0.6:
                        badge_class = "badge-good"
                    else:
                        badge_class = "badge-poor"

                    html_content += f"                            <td style=\"text-align: center;\"><span class=\"score-badge {badge_class}\">{score_str}</span></td>\n"
                else:
                    html_content += "                            <td style=\"text-align: center;\">N/A</td>\n"

            html_content += "                        </tr>\n"

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- Interprétation -->
            <div class="section">
                <h2 class="section-title">💡 Interprétation des Scores</h2>
                <div class="interpretation">
                    <h3>📊 Guide de lecture</h3>
                    <ul>
                        <li><strong>Faithfulness (Fidélité)</strong> : Mesure si la réponse est fidèle au contexte récupéré. > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer</li>
                        <li><strong>Answer Relevancy (Pertinence)</strong> : Mesure la pertinence de la réponse à la question. > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer</li>
                        <li><strong>Context Precision (Précision du contexte)</strong> : Mesure la précision du contexte récupéré. > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer</li>
                        <li><strong>Context Recall (Rappel du contexte)</strong> : Mesure la complétude du contexte récupéré. > 0.8 = Excellent | 0.6-0.8 = Bon | < 0.6 = À améliorer</li>
                    </ul>
                </div>
            </div>

            <!-- Recommandations -->
            <div class="section">
                <h2 class="section-title">🎯 Recommandations</h2>
"""

    # Générer les recommandations basées sur les scores
    has_recommendations = False

    if scores.get("faithfulness", 0) < 0.7:
        has_recommendations = True
        html_content += """
                <div class="recommendations">
                    <h3>⚠️  Faithfulness faible</h3>
                    <ul>
                        <li>Vérifiez que les réponses restent fidèles au contexte</li>
                        <li>Ajustez le prompt système pour éviter les hallucinations</li>
                    </ul>
                </div>
"""

    if scores.get("answer_relevancy", 0) < 0.7:
        has_recommendations = True
        html_content += """
                <div class="recommendations">
                    <h3>⚠️  Answer Relevancy faible</h3>
                    <ul>
                        <li>Vérifiez que les réponses adressent bien la question</li>
                        <li>Améliorez la qualité du prompt d'enrichissement</li>
                    </ul>
                </div>
"""

    if scores.get("context_precision", 0) < 0.7:
        has_recommendations = True
        html_content += """
                <div class="recommendations">
                    <h3>⚠️  Context Precision faible</h3>
                    <ul>
                        <li>Améliorez la qualité des embeddings (modèle, chunking)</li>
                        <li>Ajustez les paramètres de recherche (top_k, seuil)</li>
                    </ul>
                </div>
"""

    if scores.get("context_recall", 0) < 0.7:
        has_recommendations = True
        html_content += """
                <div class="recommendations">
                    <h3>⚠️  Context Recall faible</h3>
                    <ul>
                        <li>Augmentez le nombre de contextes récupérés (top_k)</li>
                        <li>Vérifiez la complétude de votre base de données</li>
                    </ul>
                </div>
"""

    if not has_recommendations:
        html_content += """
                <div class="success-message">
                    ✅ Tous les scores sont satisfaisants ! Votre système RAG fonctionne correctement.
                </div>
"""

    html_content += """
            </div>
        </div>

        <div class="footer">
            <p>Rapport généré automatiquement par le système d'évaluation RAGAS</p>
            <p>Projet OpenClassrooms - Événements Culturels Occitanie</p>
        </div>
    </div>
</body>
</html>
"""

    # Écrire le fichier HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n📄 Rapport HTML généré: {output_path}")


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
  # Utiliser le fichier par défaut (ragas_data/ragas_test_questions_collected.json)
  python tests/evaluate_ragas.py

  # Utiliser un fichier spécifique
  python tests/evaluate_ragas.py tests/ragas_data/ragas_test_questions_collected.json
  python tests/evaluate_ragas.py tests/ragas_data/my_custom_questions.json
        """
    )
    parser.add_argument(
        "questions_file",
        nargs="?",
        default=None,
        help="Chemin vers le fichier de questions JSON (défaut: ragas_data/ragas_test_questions_collected.json)"
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
    ragas_data = validate_ragas_data(test_questions)

    # Générer le rapport
    if ragas_data:
        generate_ragas_report(ragas_data)
    else:
        print("\n❌ Aucune donnée collectée. Impossible de générer le rapport.")
        sys.exit(1)


if __name__ == "__main__":
    main()
