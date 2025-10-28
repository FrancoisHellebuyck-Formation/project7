"""
Serveur de recherche vectorielle FAISS.

Ce module démarre un serveur qui charge le vector store FAISS en mémoire
et le garde prêt à recevoir des requêtes de recherche sémantique.
Utilise une interface simple en ligne de commande (REPL).
"""

import os
import sys
from pathlib import Path
import logging
from typing import Optional
from dotenv import load_dotenv

# Ajouter le parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from embeddings import get_embeddings_model
from vectors import (
    load_vector_store,
    search_similar_documents,
    get_vector_store_stats,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreServer:
    """
    Serveur de recherche vectorielle FAISS.

    Charge le vector store en mémoire et fournit une interface
    pour effectuer des recherches sémantiques.
    """

    def __init__(self, index_path: str, model_id: Optional[str] = None):
        """
        Initialise le serveur avec un vector store.

        Args:
            index_path: Chemin vers l'index FAISS
            model_id: Identifiant du modèle d'embeddings (optionnel)
        """
        self.index_path = index_path
        self.model_id = model_id
        self.embeddings = None
        self.vector_store = None
        self.is_loaded = False

    def start(self) -> None:
        """
        Démarre le serveur en chargeant le vector store en mémoire.
        """
        logger.info("="*70)
        logger.info("DÉMARRAGE DU SERVEUR DE RECHERCHE VECTORIELLE")
        logger.info("="*70)

        # Vérifier que l'index existe
        if not Path(self.index_path).exists():
            logger.error(f"❌ L'index FAISS n'existe pas: {self.index_path}")
            logger.info("💡 Créez d'abord l'index avec: make run-embeddings")
            return

        try:
            # 1. Charger le modèle d'embeddings
            logger.info("\n[1/2] Chargement du modèle d'embeddings...")
            self.embeddings = get_embeddings_model(model_id=self.model_id)

            # 2. Charger le vector store
            logger.info("\n[2/2] Chargement du vector store...")
            self.vector_store = load_vector_store(
                self.index_path,
                self.embeddings,
                verbose=True
            )

            # Afficher les statistiques
            stats = get_vector_store_stats(self.vector_store, verbose=True)

            self.is_loaded = True

            logger.info("\n" + "="*70)
            logger.info("✅ SERVEUR PRÊT - En attente de requêtes")
            logger.info("="*70)
            logger.info(f"📊 {stats['num_vectors']:,} vecteurs indexés")
            logger.info(f"📐 Dimension: {stats['dimension']}")
            logger.info("\nCommandes disponibles:")
            logger.info("  - Tapez votre requête pour rechercher")
            logger.info("  - 'stats' pour afficher les statistiques")
            logger.info("  - 'help' pour l'aide")
            logger.info("  - 'quit' ou 'exit' pour quitter")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage: {e}", exc_info=True)
            self.is_loaded = False

    def search(self, query: str, k: int = 5, verbose: bool = True) -> None:
        """
        Effectue une recherche sémantique.

        Args:
            query: Requête textuelle
            k: Nombre de résultats à retourner
            verbose: Si True, affiche les résultats détaillés
        """
        if not self.is_loaded:
            logger.error("❌ Le serveur n'est pas démarré. Appelez start() d'abord.")
            return

        if not query.strip():
            logger.warning("⚠️  Requête vide, veuillez entrer une recherche.")
            return

        logger.info(f"\n🔍 Recherche: '{query}'")
        logger.info("-" * 70)

        try:
            results = search_similar_documents(
                self.vector_store,
                query,
                k=k,
                verbose=False  # On gère l'affichage nous-mêmes
            )

            if not results:
                logger.info("Aucun résultat trouvé.")
                return

            logger.info(f"✅ {len(results)} résultats trouvés\n")

            for i, (doc, score) in enumerate(results, 1):
                logger.info(f"{'='*70}")
                logger.info(f"Résultat {i}/{len(results)} - Score: {score:.4f}")
                logger.info(f"{'='*70}")
                logger.info(f"📌 Titre: {doc.metadata.get('title', 'N/A')}")
                logger.info(f"📍 Lieu: {doc.metadata.get('city', 'N/A')}")
                logger.info(f"🗓️  Date: {doc.metadata.get('dateRange', 'N/A')}")
                logger.info(f"🏷️  Région: {doc.metadata.get('region', 'N/A')}")

                # Afficher les mots-clés si disponibles
                keywords = doc.metadata.get('keywords', [])
                if keywords:
                    logger.info(f"🔖 Mots-clés: {', '.join(keywords[:5])}")

                # Extraire et afficher un extrait du contenu
                content = doc.page_content.strip()
                # Prendre les 300 premiers caractères
                excerpt = content[:300] + "..." if len(content) > 300 else content
                logger.info(f"\n📄 Extrait:\n{excerpt}")
                logger.info("")

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}", exc_info=True)

    def show_stats(self) -> None:
        """Affiche les statistiques du vector store."""
        if not self.is_loaded:
            logger.error("❌ Le serveur n'est pas démarré.")
            return

        logger.info("\n" + "="*70)
        logger.info("📊 STATISTIQUES DU VECTOR STORE")
        logger.info("="*70)
        stats = get_vector_store_stats(self.vector_store, verbose=True)
        logger.info(f"📂 Chemin: {self.index_path}")
        logger.info(f"🤖 Modèle: {self.model_id or 'intfloat/multilingual-e5-large'}")
        logger.info("="*70 + "\n")

    def run_repl(self) -> None:
        """
        Lance une boucle REPL (Read-Eval-Print-Loop) pour interagir avec le serveur.
        """
        if not self.is_loaded:
            logger.error("❌ Le serveur n'est pas démarré. Appelez start() d'abord.")
            return

        logger.info("\n💬 Mode interactif activé")

        while True:
            try:
                # Lire l'entrée utilisateur
                user_input = input("\n🔍 Recherche> ").strip()

                if not user_input:
                    continue

                # Commandes spéciales
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("\n👋 Arrêt du serveur...")
                    break

                elif user_input.lower() == 'stats':
                    self.show_stats()

                elif user_input.lower() == 'help':
                    self.show_help()

                elif user_input.lower().startswith('top'):
                    # Permet de spécifier le nombre de résultats: "top 10"
                    parts = user_input.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        k = int(parts[1])
                        logger.info(f"Mode modifié: top {k} résultats par défaut")
                    else:
                        logger.info("Usage: top <nombre>")

                else:
                    # Effectuer la recherche
                    self.search(user_input, k=5)

            except KeyboardInterrupt:
                logger.info("\n\n👋 Arrêt du serveur (Ctrl+C)...")
                break

            except EOFError:
                logger.info("\n\n👋 Arrêt du serveur (EOF)...")
                break

            except Exception as e:
                logger.error(f"❌ Erreur: {e}", exc_info=True)

    def show_help(self) -> None:
        """Affiche l'aide."""
        logger.info("\n" + "="*70)
        logger.info("📖 AIDE - SERVEUR DE RECHERCHE VECTORIELLE")
        logger.info("="*70)
        logger.info("""
Commandes disponibles:

  <votre recherche>    Effectue une recherche sémantique
                       Exemple: "concert de jazz à Toulouse"

  stats               Affiche les statistiques du vector store

  help                Affiche cette aide

  quit / exit / q     Quitte le serveur

Exemples de recherches:
  - "exposition d'art contemporain"
  - "spectacle pour enfants"
  - "festival de musique électronique"
  - "conférence sur l'environnement"
  - "marché de Noël"
        """)
        logger.info("="*70)


def main():
    """
    Point d'entrée principal du serveur.
    """
    # Charger les variables d'environnement
    load_dotenv()

    # Configuration
    index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    model_id = os.getenv("EMBEDDINGS_MODEL")

    # Créer et démarrer le serveur
    server = VectorStoreServer(index_path, model_id)
    server.start()

    # Si le chargement a réussi, lancer le REPL
    if server.is_loaded:
        try:
            server.run_repl()
        except Exception as e:
            logger.error(f"❌ Erreur dans le REPL: {e}", exc_info=True)
    else:
        logger.error("❌ Impossible de démarrer le serveur")
        sys.exit(1)


if __name__ == "__main__":
    main()
