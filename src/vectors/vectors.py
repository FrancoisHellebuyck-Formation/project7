"""
Module pour la gestion de la base vectorielle FAISS.

Ce module gère la création, sauvegarde, chargement et recherche dans l'index FAISS.
Responsabilité unique : opérations sur les vector stores.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    verbose: bool = False
) -> FAISS:
    """
    Crée un vector store FAISS à partir des documents.

    Args:
        documents: Liste de documents LangChain à vectoriser
        embeddings: Modèle d'embeddings à utiliser
        verbose: Si True, affiche des informations de progression

    Returns:
        FAISS: Instance du vector store créé

    Raises:
        ValueError: Si la liste de documents est vide
    """
    if not documents:
        raise ValueError("La liste de documents est vide")

    if verbose:
        logger.info(f"Création du vector store pour {len(documents)} documents...")
        logger.info("Génération des embeddings (cela peut prendre du temps)...")

    # Créer le vector store FAISS
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    if verbose:
        logger.info(f"✓ Vector store créé avec {len(documents)} vecteurs")

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    save_path: str,
    verbose: bool = False
) -> None:
    """
    Sauvegarde le vector store FAISS sur le disque.

    Args:
        vector_store: Instance du vector store FAISS
        save_path: Chemin du répertoire de sauvegarde
        verbose: Si True, affiche des informations de progression
    """
    # Créer le répertoire s'il n'existe pas
    Path(save_path).mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"Sauvegarde du vector store dans: {save_path}")

    vector_store.save_local(save_path)

    if verbose:
        logger.info("✓ Vector store sauvegardé avec succès")


def load_vector_store(
    load_path: str,
    embeddings: Embeddings,
    verbose: bool = False
) -> FAISS:
    """
    Charge un vector store FAISS depuis le disque.

    Args:
        load_path: Chemin du répertoire contenant le vector store
        embeddings: Modèle d'embeddings (doit être le même que lors de la création)
        verbose: Si True, affiche des informations de progression

    Returns:
        FAISS: Instance du vector store chargé

    Raises:
        FileNotFoundError: Si le répertoire n'existe pas
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Le répertoire {load_path} n'existe pas")

    if verbose:
        logger.info(f"Chargement du vector store depuis: {load_path}")

    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    if verbose:
        logger.info("✓ Vector store chargé avec succès")

    return vector_store


def search_similar_documents(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    verbose: bool = False
) -> List[Tuple[Document, float]]:
    """
    Recherche les documents les plus similaires à une requête.

    Args:
        vector_store: Instance du vector store FAISS
        query: Requête textuelle
        k: Nombre de résultats à retourner
        verbose: Si True, affiche des informations de progression

    Returns:
        list: Liste de tuples (Document, score de similarité)
    """
    if verbose:
        logger.info(f"Recherche de {k} documents similaires à: '{query[:50]}...'")

    results = vector_store.similarity_search_with_score(query, k=k)

    if verbose:
        logger.info(f"✓ {len(results)} résultats trouvés")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"  {i}. Score: {score:.4f} | {doc.metadata.get('title', 'Sans titre')[:60]}")

    return results


def add_documents_to_vector_store(
    vector_store: FAISS,
    documents: List[Document],
    verbose: bool = False
) -> FAISS:
    """
    Ajoute de nouveaux documents à un vector store existant.

    Args:
        vector_store: Instance du vector store FAISS existant
        documents: Liste de documents à ajouter
        verbose: Si True, affiche des informations de progression

    Returns:
        FAISS: Instance du vector store mise à jour
    """
    if not documents:
        logger.warning("Aucun document à ajouter")
        return vector_store

    if verbose:
        logger.info(f"Ajout de {len(documents)} documents au vector store...")

    vector_store.add_documents(documents)

    if verbose:
        logger.info(f"✓ {len(documents)} documents ajoutés avec succès")

    return vector_store


def delete_vector_store(
    path: str,
    verbose: bool = False
) -> None:
    """
    Supprime un vector store du disque.

    Args:
        path: Chemin du répertoire contenant le vector store
        verbose: Si True, affiche des informations de progression

    Note:
        Si le répertoire n'existe pas, affiche un warning mais ne lève pas d'exception
    """
    import shutil

    path_obj = Path(path)
    if not path_obj.exists():
        if verbose:
            logger.warning(f"⚠️  Le répertoire {path} n'existe pas (aucune suppression nécessaire)")
        return

    if verbose:
        logger.info(f"Suppression du vector store: {path}")

    shutil.rmtree(path)

    if verbose:
        logger.info("✓ Vector store supprimé avec succès")


def get_vector_store_stats(
    vector_store: FAISS,
    verbose: bool = False
) -> dict:
    """
    Récupère les statistiques d'un vector store.

    Args:
        vector_store: Instance du vector store FAISS
        verbose: Si True, affiche les statistiques

    Returns:
        dict: Dictionnaire contenant les statistiques
    """
    stats = {
        "num_vectors": vector_store.index.ntotal,
        "dimension": vector_store.index.d,
    }

    if verbose:
        logger.info(f"Statistiques du vector store:")
        logger.info(f"  - Nombre de vecteurs: {stats['num_vectors']}")
        logger.info(f"  - Dimension: {stats['dimension']}")

    return stats


def main():
    """
    Fonction de test pour les opérations de base sur les vector stores.
    Teste le chargement, la recherche et les statistiques.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logger.info("="*70)
    logger.info("TEST DU MODULE VECTORS")
    logger.info("="*70)

    # Configuration
    index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

    # Vérifier si un index existe
    if not Path(index_path).exists():
        logger.warning(f"\nAucun index trouvé à {index_path}")
        logger.info("Exécutez d'abord 'make run-embeddings' pour créer un index.")
        return

    try:
        # Charger le modèle d'embeddings
        logger.info("\n1. Chargement du modèle d'embeddings...")
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from embeddings import get_embeddings_model

        embeddings = get_embeddings_model()

        # Charger l'index
        logger.info("\n2. Chargement du vector store...")
        vector_store = load_vector_store(index_path, embeddings, verbose=True)

        # Afficher les statistiques
        logger.info("\n3. Statistiques du vector store:")
        stats = get_vector_store_stats(vector_store, verbose=True)

        # Test de recherche
        test_query = os.getenv("TEST_QUERY", "concert de musique")
        logger.info(f"\n4. Test de recherche avec: '{test_query}'")
        results = search_similar_documents(vector_store, test_query, k=3, verbose=True)

        logger.info("\n5. Résultats détaillés:")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"\n   Résultat {i} (Score: {score:.4f})")
            logger.info(f"   Titre: {doc.metadata.get('title', 'N/A')}")
            logger.info(f"   Lieu: {doc.metadata.get('city', 'N/A')}")
            logger.info(f"   Extrait: {doc.page_content[:150]}...")

        logger.info("\n" + "="*70)
        logger.info("✓ TEST TERMINÉ AVEC SUCCÈS")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
