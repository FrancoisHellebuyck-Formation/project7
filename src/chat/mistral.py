import os
import logging
import requests
from mistralai import Mistral, UserMessage, SystemMessage
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Configuration et Initialisation ---

# Charger les variables d'environnement (y compris MISTRAL_API_KEY)
load_dotenv()

# Récupérer la clé API.
# Assurez-vous d'avoir la variable MISTRAL_API_KEY définie dans votre environnement ou dans un fichier .env
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "La variable d'environnement MISTRAL_API_KEY n'est pas définie. Veuillez la configurer."
    )

# Initialisation du client Mistral
client = Mistral(api_key=api_key)

# Définition du modèle à utiliser
# Modèles courants : 'mistral-small-latest', 'mistral-large-latest', 'open-mixtral-8x7b'
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_TEMPERATURE = float(os.getenv("MISTRAL_TEMPERATURE", "0.7"))

# Configuration de l'API RAG
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_API_SEARCH_ENDPOINT = f"{RAG_API_URL}/search"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))


# --- 2. Fonction de Recherche RAG ---
def search_rag(query: str, k: int = RAG_TOP_K) -> list:
    """
    Effectue une recherche dans le vector store via l'API RAG.

    Args:
        query (str): La requête de recherche
        k (int): Nombre de résultats à retourner

    Returns:
        list: Liste des résultats de recherche avec leurs métadonnées
    """
    try:
        logger.info(f"Recherche RAG pour: '{query}' (k={k})")

        response = requests.post(
            RAG_API_SEARCH_ENDPOINT, json={"query": query, "k": k}, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        logger.info(f"✓ {len(results)} résultats trouvés")
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors de l'appel à l'API RAG: {e}")
        return []


def format_rag_context(results: list) -> str:
    """
    Formate les résultats RAG en contexte texte pour enrichir le prompt.

    Args:
        results (list): Liste des résultats de recherche

    Returns:
        str: Contexte formaté
    """
    if not results:
        return "Aucune information contextuelle trouvée."

    context_parts = [
        "Voici les informations pertinentes trouvées dans la base de données:\n"
    ]

    for i, result in enumerate(results, 1):
        title = result.get("title", "Sans titre")
        content = result.get("content", "")
        score = result.get("score", 0)
        metadata = result.get("metadata", {})

        # Limiter la longueur du contenu pour ne pas surcharger le prompt
        content_preview = content[:500] + "..." if len(content) > 500 else content

        context_parts.append(f"\n--- Résultat {i} (pertinence: {score:.3f}) ---")
        context_parts.append(f"Titre: {title}")

        if metadata.get("city"):
            context_parts.append(f"Ville: {metadata['city']}")
        if metadata.get("date_debut"):
            context_parts.append(f"Date début: {metadata['date_debut']}")
        if metadata.get("date_fin"):
            context_parts.append(f"Date fin: {metadata['date_fin']}")

        context_parts.append(f"\nContenu:\n{content_preview}")

    return "\n".join(context_parts)


# --- 3. Définition des Messages (Conversation) ---
def get_system_prompt(chemin_fichier):
    """
    Lit le contenu complet du prompt système depuis un fichier .md.

    Args:
        chemin_fichier (str): Le chemin vers le fichier .md.
    """
    try:
        # Utiliser 'with open' est la meilleure pratique :
        # cela garantit que le fichier est correctement fermé, même en cas d'erreur.
        # 'r' est pour la lecture (read), 'utf-8' gère les caractères spéciaux (é, à, ü, etc.).
        with open(chemin_fichier, "r", encoding="utf-8") as f:
            # Lire tout le contenu du fichier
            r = f.read()
            return r
    except FileNotFoundError:
        return f"Erreur : Le fichier spécifié '{chemin_fichier}' est introuvable."
    except Exception as e:
        return f"Une erreur inattendue s'est produite : {e}"


# Le format utilise une liste de dictionnaires (objets ChatMessage)
# Chaque message a un 'role' (system, user, assistant) et un 'content'

systemMessage_content = get_system_prompt(
    os.path.join(os.path.dirname(__file__), "ps.md")
)

# Question de l'utilisateur
user_question = (
    "Quel est le festival de musique le plus célèbre de la région Occitanie en été ?"
)

# --- 4. Enrichissement avec RAG ---

logger.info("=" * 70)
logger.info("ENRICHISSEMENT DU PROMPT AVEC RAG")
logger.info("=" * 70)

# Recherche d'informations contextuelles via l'API RAG
rag_results = search_rag(user_question, k=RAG_TOP_K)

# Formatage du contexte
rag_context = format_rag_context(rag_results)

# Construction du prompt enrichi
enriched_user_prompt = f"""{rag_context}

---

Question de l'utilisateur:
{user_question}

Réponds à la question en te basant sur les informations contextuelles ci-dessus. Si les informations ne permettent pas de répondre complètement, indique-le clairement."""

logger.info("\n" + "=" * 70)
logger.info("PROMPT ENRICHI CONSTRUIT")
logger.info("=" * 70)

messages = [
    # 1. Le rôle 'system' sert à définir le comportement de l'IA (votre prompt système)
    SystemMessage(
        content=systemMessage_content,
        role="system",
    ),
    # 2. Le message de l'utilisateur enrichi avec le contexte RAG
    UserMessage(
        role="user",
        content=enriched_user_prompt,
    ),
]

# --- 5. Appel de l'API ---

logger.info(f"\n💬 Requête envoyée au modèle : {MODEL_NAME}...")

try:
    # Appel de la méthode de complétion de chat
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=messages,
        temperature=MISTRAL_TEMPERATURE
    )

    # --- 6. Affichage du Résultat ---

    # Le contenu de la réponse se trouve dans le premier choix de la liste 'choices'
    response_content = response.choices[0].message.content

    logger.info("\n" + "=" * 70)
    logger.info("RÉPONSE DE MISTRAL AI")
    logger.info("=" * 70)
    print(f"\n{response_content.strip()}\n")
    logger.info("=" * 70)

    # Affichage des métriques d'utilisation (optionnel)
    logger.info("📊 Utilisation des jetons (tokens) :")
    logger.info(f"  - Entrée : {response.usage.prompt_tokens}")
    logger.info(f"  - Sortie : {response.usage.completion_tokens}")
    logger.info(f"  - Total : {response.usage.total_tokens}")


except Exception as e:
    logger.error(f"❌ Une erreur s'est produite lors de l'appel API : {e}")
