"""
Script utilitaire pour pré-collecter les données answer et contexts.

Ce script lit ragas_test_questions.json, interroge l'API RAG pour chaque
question, et génère un nouveau fichier JSON avec les réponses et contextes
pré-collectés. Cela permet d'éviter de réinterroger l'API lors de chaque
évaluation RAGAS.

Usage:
    python tests/collect_ragas_data.py

Le fichier de sortie sera sauvegardé dans tests/ragas_test_questions_collected.json
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

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
RAGAS_MISTRAL_RETRY = int(os.getenv("RAGAS_MISTRAL_RETRY", "3"))


def check_api_health() -> bool:
    """Vérifie que l'API RAG est accessible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return (
                data.get("status") in ["ok", "healthy"]
                and data.get("vector_store_loaded", False)
                and data.get("embeddings_model_loaded", False)
            )
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors du health check: {e}")
        return False


def get_rag_response(question: str) -> dict:
    """Récupère une réponse du système RAG via l'API."""
    for attempt in range(1, RAGAS_MISTRAL_RETRY + 1):
        try:
            # Appliquer le délai AVANT l'appel API (éviter rate limiting)
            if RAGAS_MISTRAL_DELAY > 0:
                if attempt == 1:
                    print(
                        f"   ⏳ Attente de {RAGAS_MISTRAL_DELAY}s "
                        "(délai anti-rate-limiting)..."
                    )
                else:
                    print(
                        f"   ⏳ Retry {attempt}/{RAGAS_MISTRAL_RETRY} - "
                        f"Attente de {RAGAS_MISTRAL_DELAY}s..."
                    )
                time.sleep(RAGAS_MISTRAL_DELAY)

            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "k": RAGAS_TOP_K},
                timeout=RAGAS_API_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            return result
        except requests.exceptions.HTTPError as e:
            is_429_error = False

            if e.response.status_code == 429:
                # Erreur 429 directe
                is_429_error = True
                print(
                    f"   ⚠️  Erreur 429 (rate limiting) - "
                    f"Tentative {attempt}/{RAGAS_MISTRAL_RETRY}"
                )
            elif e.response.status_code == 500:
                # Vérifier si c'est un 429 encapsulé dans un 500
                try:
                    error_detail = e.response.json().get("detail", "")
                    if (
                        "429" in error_detail
                        or "capacity exceeded" in error_detail.lower()
                    ):
                        is_429_error = True
                        print(
                            f"   ⚠️  Quota API dépassé (429 encapsulé) - "
                            f"Tentative {attempt}/{RAGAS_MISTRAL_RETRY}"
                        )
                except Exception:
                    pass

            # Si c'est une erreur 429 et qu'il reste des tentatives, retry
            if is_429_error and attempt < RAGAS_MISTRAL_RETRY:
                continue

            # Sinon, afficher l'erreur et retourner None
            if not is_429_error:
                print(f"   ❌ Erreur HTTP: {e}")
            else:
                print(f"   ❌ Échec après {RAGAS_MISTRAL_RETRY} tentatives")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erreur de requête: {e}")
            return None

    return None


def deduplicate_contexts(context_used: list) -> list:
    """
    Déduplique les contextes en se basant sur le contenu complet.

    Supprime les contextes qui ont exactement le même contenu textuel.
    Cela permet d'éviter les doublons parfaits tout en gardant des contextes
    différents d'un même événement.

    Args:
        context_used: Liste des contextes retournés par l'API RAG

    Returns:
        Liste de contextes dédupliqués (conserve l'ordre)
    """
    seen_contents = set()
    deduplicated = []

    for ctx in context_used:
        content = ctx.get("content", "")
        # Normaliser le contenu pour la comparaison (enlever espaces superflus)
        normalized_content = " ".join(content.split())

        if normalized_content and normalized_content not in seen_contents:
            seen_contents.add(normalized_content)
            deduplicated.append(ctx)

    return deduplicated


def format_contexts(context_used: list) -> list:
    """
    Formate les contextes pour RAGAS (liste de strings).

    Utilise le contenu de l'événement pour l'évaluation RAGAS.
    Cela permet à RAGAS d'évaluer la pertinence basée sur le contenu
    textuel complet des événements récupérés.

    Args:
        context_used: Liste des contextes retournés par l'API RAG

    Returns:
        Liste de contenus d'événements (strings)
    """
    return [ctx.get("content", "") for ctx in context_used]


def generate_ground_truth(context_used: list, category: str) -> str:
    """
    Génère un ground_truth basé sur les métadonnées des événements.

    Args:
        context_used: Liste des contextes retournés par l'API RAG
        category: Catégorie du test case

    Returns:
        Ground truth formaté listant les titres des événements
    """
    # Cas spécial: question hors sujet
    if category == "non connu":
        return "Je suis désolé, mais je ne peux pas vous fournir *information*, car cette information n'est pas disponible dans les données contextuelles fournies. Mon expertise se limite aux événements culturels en Occitanie."

    # Extraire les titres des événements depuis les métadonnées
    event_titles = []
    for ctx in context_used:
        metadata = ctx.get("metadata", {})
        title = metadata.get("title")
        if title:
            event_titles.append(title)

    # Construire le ground_truth
    if not event_titles:
        return "Aucun événement trouvé dans les documents."

    if len(event_titles) == 1:
        return f"L'événement disponible est : {event_titles[0]}."

    # Plusieurs événements
    intro = "Parmi les événements disponibles : "
    event_list = ", ".join([f"{i}. {title}" for i, title in enumerate(event_titles, 1)])
    return intro + event_list + "."


def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 70)
    print("🔧 COLLECTE DES DONNÉES RAGAS")
    print("=" * 70)
    print("\nCe script interroge l'API RAG pour chaque question et sauvegarde")
    print("les réponses et contextes dans un nouveau fichier JSON.")
    print("")

    # Vérifier l'API
    print("🔍 Vérification de l'API RAG...")
    if not check_api_health():
        print("\n❌ L'API RAG n'est pas accessible")
        print(f"   URL: {API_URL}")
        print("   Démarrez l'API avec: make run-api")
        sys.exit(1)

    print("✅ API RAG accessible\n")

    # Charger le fichier JSON
    input_path = Path(__file__).parent / "ragas_test_questions.json"
    if not input_path.exists():
        print(f"❌ Fichier non trouvé: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if not test_cases:
        print("❌ Aucun cas de test trouvé dans le JSON")
        sys.exit(1)

    print(f"📋 {len(test_cases)} cas de test trouvés")
    print(
        f"⚙️  Configuration: top_k={RAGAS_TOP_K}, "
        f"delay={RAGAS_MISTRAL_DELAY}s, retry={RAGAS_MISTRAL_RETRY}"
    )
    print("")

    # Collecter les données
    collected = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case.get("id", f"test_{i:03d}")
        question = test_case["question"]

        print(f"[{i}/{len(test_cases)}] {test_id}")
        print(f"   Question: {question}")

        # Vérifier si déjà collecté
        if test_case.get("answer") and test_case.get("contexts"):
            print("   ⏭️  Déjà collecté, passage au suivant")
            collected += 1
            continue

        # Interroger l'API
        print("   🔍 Interrogation de l'API RAG...")
        response = get_rag_response(question)

        if response is None:
            print("   ❌ Échec de la collecte")
            failed += 1
            continue

        # Vérifier la structure
        if "answer" not in response or "context_used" not in response:
            print("   ❌ Réponse invalide (structure incorrecte)")
            failed += 1
            continue

        answer = response["answer"]
        contexts = response["context_used"]

        # Valider
        if not answer or len(answer) < 50:
            print(f"   ⚠️  Réponse trop courte ({len(answer)} caractères)")
            failed += 1
            continue

        if len(contexts) == 0:
            print("   ⚠️  Aucun contexte récupéré")
            failed += 1
            continue

        # Dédupliquer les contextes (supprime les contenus totalement identiques)
        original_count = len(contexts)
        contexts = deduplicate_contexts(contexts)
        deduplicated_count = len(contexts)

        if original_count != deduplicated_count:
            removed_count = original_count - deduplicated_count
            print(
                f"   🔄 Déduplication: {removed_count} doublon(s) parfait(s) "
                f"supprimé(s) ({deduplicated_count} restants)"
            )

        # Générer le ground_truth basé sur les métadonnées
        # Si ground_truth existe déjà dans le fichier original, on le conserve
        existing_ground_truth = test_case.get("ground_truth")
        if existing_ground_truth and existing_ground_truth.strip():
            # Utiliser le ground_truth existant
            ground_truth = existing_ground_truth
            print("   📝 Ground truth: conservé du fichier original")
        else:
            # Générer automatiquement le ground_truth
            category = test_case.get("category", "")
            ground_truth = generate_ground_truth(contexts, category)
            print("   📝 Ground truth: généré automatiquement")
            print(f"      Aperçu: {ground_truth[:80]}...")

        # Sauvegarder dans le cas de test
        test_case["answer"] = answer
        test_case["contexts"] = format_contexts(contexts)
        test_case["ground_truth"] = ground_truth

        print(
            f"   ✅ Collecté: {len(answer)} caractères, "
            f"{deduplicated_count} contextes"
        )
        collected += 1

    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"  ✅ Collectés: {collected}/{len(test_cases)}")
    print(f"  ❌ Échecs: {failed}/{len(test_cases)}")
    print("")

    if collected == 0:
        print("❌ Aucune donnée collectée. Abandon de la sauvegarde.")
        sys.exit(1)

    # Mettre à jour les métadonnées
    if "metadata" not in data:
        data["metadata"] = {}

    data["metadata"]["last_collected_at"] = datetime.now().isoformat()
    data["metadata"]["collected_count"] = collected
    data["metadata"]["failed_count"] = failed

    # Sauvegarder
    output_path = Path(__file__).parent / "ragas_test_questions_collected.json"

    print(f"💾 Sauvegarde dans: {output_path.name}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ Fichier sauvegardé avec succès")
    print("")
    print("📝 Prochaine étape:")
    print("   Lancez l'évaluation RAGAS: make test-ragas")
    print("")
    print("ℹ️  Note: Le script d'évaluation utilisera automatiquement")
    print(f"   {output_path.name}")
    print("")
    print("=" * 70)


if __name__ == "__main__":
    main()
