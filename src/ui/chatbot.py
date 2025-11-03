"""
Chatbot Streamlit pour les événements culturels en Occitanie.

Ce module fournit une interface utilisateur web interactive pour poser
des questions sur les événements culturels en utilisant l'API RAG.
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

import streamlit as st
import requests
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Configuration
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
ASK_ENDPOINT = f"{API_URL}/ask"
DEFAULT_K = int(os.getenv("RAG_TOP_K", "5"))


# ============================================================================
# Configuration de la page Streamlit
# ============================================================================

st.set_page_config(
    page_title="Puls-Events Chatbot",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Fonctions utilitaires
# ============================================================================


def init_session_state():
    """Initialise l'état de session Streamlit pour stocker l'historique."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False


def call_ask_api(question: str, k: int = DEFAULT_K) -> Optional[Dict]:
    """
    Appelle l'endpoint /ask de l'API.

    Args:
        question: Question de l'utilisateur
        k: Nombre de documents contextuels à récupérer

    Returns:
        Réponse de l'API au format dict, ou None en cas d'erreur
    """
    try:
        logger.info(f"Appel API /ask avec question: '{question}' (k={k})")

        response = requests.post(
            ASK_ENDPOINT, json={"question": question, "k": k}, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        logger.info(
            f"Réponse reçue - tokens utilisés: {data['tokens_used']['total_tokens']}"
        )

        return data

    except requests.exceptions.ConnectionError:
        logger.error("Impossible de se connecter à l'API")
        st.error(
            "❌ Impossible de se connecter à l'API. Vérifiez que le serveur est démarré."
        )
        return None

    except requests.exceptions.Timeout:
        logger.error("Timeout de la requête API")
        st.error("⏱️ La requête a pris trop de temps. Réessayez.")
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"Erreur HTTP: {e}")
        st.error(f"❌ Erreur API: {e}")
        return None

    except Exception as e:
        logger.error(f"Erreur inattendue: {e}", exc_info=True)
        st.error(f"❌ Erreur inattendue: {e}")
        return None


def format_context_sources(context_used: List[Dict]) -> str:
    """
    Formate les sources contextuelles pour l'affichage.

    Args:
        context_used: Liste des documents utilisés comme contexte

    Returns:
        String formaté avec les sources
    """
    if not context_used:
        return "Aucune source"

    sources = []
    for i, ctx in enumerate(context_used[:3], 1):  # Limite à 3 sources
        title = ctx.get("title", "Sans titre")
        score = ctx.get("score", 0)
        city = ctx.get("metadata", {}).get("city", "N/A")
        sources.append(f"{i}. **{title}** ({city}) - Score: {score:.3f}")

    if len(context_used) > 3:
        sources.append(f"... et {len(context_used) - 3} autre(s) document(s)")

    return "\n".join(sources)


def add_message(role: str, content: str, metadata: Optional[Dict] = None):
    """
    Ajoute un message à l'historique.

    Args:
        role: 'user' ou 'assistant'
        content: Contenu du message
        metadata: Métadonnées optionnelles (tokens, sources, etc.)
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    st.session_state.messages.append(message)


def display_chat_message(message: Dict):
    """
    Affiche un message du chat avec son avatar.

    Args:
        message: Dictionnaire contenant le message
    """
    role = message["role"]
    content = message["content"]
    metadata = message.get("metadata", {})

    # Choix de l'avatar
    avatar = "🧑" if role == "user" else "🎭"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

        # Affichage des métadonnées pour les réponses de l'assistant
        if role == "assistant" and metadata:
            with st.expander("📊 Détails de la réponse", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    if "tokens_used" in metadata:
                        tokens = metadata["tokens_used"]
                        st.metric("Tokens utilisés", tokens.get("total_tokens", 0))
                        st.caption(
                            f"Prompt: {tokens.get('prompt_tokens', 0)} | "
                            f"Réponse: {tokens.get('completion_tokens', 0)}"
                        )

                with col2:
                    if "context_count" in metadata:
                        st.metric("Documents contextuels", metadata["context_count"])

                if "sources" in metadata:
                    st.markdown("**📚 Sources utilisées:**")
                    st.markdown(metadata["sources"])


def clear_conversation():
    """Efface l'historique de conversation."""
    st.session_state.messages = []
    st.session_state.conversation_started = False
    logger.info("Conversation effacée")


# ============================================================================
# Interface utilisateur principale
# ============================================================================


def main():
    """Point d'entrée principal de l'application Streamlit."""

    # Initialisation
    init_session_state()

    # En-tête
    st.title("🎭 Puls-Events Chatbot")
    st.markdown("Votre guide culturel pour les événements en **Occitanie**")
    st.divider()

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")

        # Paramètre k (nombre de contextes)
        k_value = st.slider(
            "Nombre de documents contextuels",
            min_value=1,
            max_value=10,
            value=DEFAULT_K,
            help="Nombre de documents à récupérer pour enrichir le contexte",
        )

        st.divider()

        # Informations API
        st.subheader("🔗 API")
        api_status = "🟢 Connectée" if check_api_connection() else "🔴 Déconnectée"
        st.markdown(f"**Statut:** {api_status}")
        st.caption(f"URL: {API_URL}")

        st.divider()

        # Statistiques de conversation
        st.subheader("📊 Statistiques")
        total_messages = len(st.session_state.messages)
        user_messages = sum(
            1 for msg in st.session_state.messages if msg["role"] == "user"
        )
        st.metric("Messages totaux", total_messages)
        st.metric("Questions posées", user_messages)

        st.divider()

        # Bouton pour effacer la conversation
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            clear_conversation()
            st.rerun()

        st.divider()

        # Informations
        with st.expander("ℹ️ À propos"):
            st.markdown(
                """
            **Puls-Events Chatbot** utilise :
            - 🔍 Recherche sémantique (FAISS)
            - 🤖 Mistral AI pour les réponses
            - 📚 Base de données d'événements culturels

            **Région couverte :** Occitanie uniquement
            """
            )

    # Zone de chat principale

    # Message de bienvenue si première visite
    if (
        not st.session_state.conversation_started
        and len(st.session_state.messages) == 0
    ):
        with st.chat_message("assistant", avatar="🎭"):
            st.markdown(
                """
            👋 Bonjour ! Je suis **Puls-Events**, votre guide culturel pour l'Occitanie.

            Je peux vous aider à trouver :
            - 🎵 Festivals de musique
            - 🎨 Expositions d'art
            - 🎭 Spectacles et théâtre
            - 📚 Événements culturels divers

            **Posez-moi une question pour commencer !**
            """
            )

    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        display_chat_message(message)

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question sur les événements culturels..."):
        # Marquer la conversation comme commencée
        st.session_state.conversation_started = True

        # Ajout du message utilisateur
        add_message("user", prompt)

        # Affichage du message utilisateur
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Appel de l'API et affichage de la réponse
        with st.chat_message("assistant", avatar="🎭"):
            with st.spinner("🤔 Recherche en cours..."):
                result = call_ask_api(prompt, k=k_value)

            if result:
                answer = result.get(
                    "answer", "Désolé, je n'ai pas pu générer de réponse."
                )
                context_used = result.get("context_used", [])
                tokens_used = result.get("tokens_used", {})

                # Affichage de la réponse
                st.markdown(answer)

                # Préparation des métadonnées
                metadata = {
                    "tokens_used": tokens_used,
                    "context_count": len(context_used),
                    "sources": format_context_sources(context_used),
                }

                # Affichage des détails
                with st.expander("📊 Détails de la réponse", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Tokens utilisés", tokens_used.get("total_tokens", 0))
                        st.caption(
                            f"Prompt: {tokens_used.get('prompt_tokens', 0)} | "
                            f"Réponse: {tokens_used.get('completion_tokens', 0)}"
                        )

                    with col2:
                        st.metric("Documents contextuels", len(context_used))

                    st.markdown("**📚 Sources utilisées:**")
                    st.markdown(metadata["sources"])

                # Ajout à l'historique
                add_message("assistant", answer, metadata)
            else:
                error_msg = (
                    "❌ Une erreur s'est produite lors de la communication avec l'API."
                )
                st.error(error_msg)
                add_message("assistant", error_msg)


def check_api_connection() -> bool:
    """
    Vérifie si l'API est accessible.

    Returns:
        True si l'API répond, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# ============================================================================
# Point d'entrée
# ============================================================================

if __name__ == "__main__":
    main()
