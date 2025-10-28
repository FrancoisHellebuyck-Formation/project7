"""
Module pour la génération d'embeddings avec le modèle multilingual-e5-large.

Ce module fournit une classe d'embeddings compatible LangChain pour générer
des vecteurs d'embeddings multilingues de haute qualité.
Utilise le modèle intfloat/multilingual-e5-large avec average pooling.
"""

from typing import List, Optional
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from langchain_core.embeddings import Embeddings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E5Embeddings(Embeddings):
    """
    Classe d'embeddings personnalisée pour le modèle multilingual-e5-large.
    Compatible avec LangChain pour une intégration transparente.

    Caractéristiques:
    - Modèle: intfloat/multilingual-e5-large
    - Dimension: 1024
    - Support multilingue (100+ langues)
    - Average pooling avec masking
    - Normalisation L2
    - Traitement par batch
    """

    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-large",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialise le modèle E5 pour les embeddings.

        Args:
            model_id: Identifiant du modèle HuggingFace
            device: Device à utiliser ('cuda', 'mps', 'cpu'). Auto-détecté si None
            batch_size: Taille des batchs pour le traitement
            max_length: Longueur maximale des séquences
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length

        # Détection automatique du device
        # Traiter les chaînes vides comme None
        if not device or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Initialisation du modèle {model_id} sur {self.device}")

        # Chargement du modèle et du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Modèle chargé avec succès (dimension: 1024)")

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applique un pooling moyen sur les hidden states en tenant compte du masque d'attention.

        Cette méthode implémente la stratégie de pooling recommandée pour E5:
        - Masque les tokens de padding pour ne pas les inclure dans la moyenne
        - Calcule la somme des vecteurs de tokens valides
        - Divise par le nombre de tokens non-padding pour obtenir la moyenne

        Args:
            last_hidden_states: Hidden states du dernier layer [batch, seq_len, hidden_dim]
            attention_mask: Masque d'attention [batch, seq_len]

        Returns:
            torch.Tensor: Embeddings moyennés [batch, hidden_dim]
        """
        # Masquer les tokens de padding (remplace par 0.0)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # Somme des vecteurs (sans padding)
        sum_embeddings = last_hidden.sum(dim=1)
        # Nombre de tokens non-padding
        num_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
        # Moyenne = Somme / Nombre de tokens
        return sum_embeddings / num_tokens

    def _embed_texts(self, texts: List[str], prefix: str = "passage: ") -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.

        Le modèle E5 requiert l'ajout de préfixes spécifiques:
        - "passage: " pour les documents à indexer
        - "query: " pour les requêtes de recherche

        Args:
            texts: Liste de textes à encoder
            prefix: Préfixe à ajouter aux textes (requis pour E5)

        Returns:
            np.ndarray: Matrice numpy des embeddings [n_texts, embedding_dim]
        """
        # Ajouter le préfixe requis par E5
        prefixed_texts = [f"{prefix}{text}" for text in texts]

        all_embeddings = []

        # Traiter par batch pour l'efficacité
        for i in range(0, len(prefixed_texts), self.batch_size):
            batch_texts = prefixed_texts[i:i + self.batch_size]

            # Tokenisation
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            # Déplacer les tenseurs sur le device approprié
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            # Calcul des embeddings (sans calcul de gradient)
            with torch.no_grad():
                outputs = self.model(**batch_dict)

            # Pooling moyen avec masking
            embeddings = self.average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )

            # Normalisation L2 (recommandé pour la similarité cosinus)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Déplacer sur CPU et convertir en numpy
            all_embeddings.append(embeddings.cpu().numpy())

        # Concaténer tous les batches
        return np.vstack(all_embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode une liste de documents (interface LangChain).

        Utilise le préfixe "passage: " pour les documents.

        Args:
            texts: Liste de textes à encoder

        Returns:
            List[List[float]]: Liste d'embeddings (liste de listes de floats)
        """
        embeddings = self._embed_texts(texts, prefix="passage: ")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Encode une requête de recherche (interface LangChain).

        Utilise le préfixe "query: " pour les requêtes (différent des documents).
        Cette distinction améliore la qualité de la recherche sémantique.

        Args:
            text: Texte de la requête

        Returns:
            List[float]: Embedding de la requête (liste de floats)
        """
        # Pour les requêtes, E5 recommande le préfixe "query: "
        embeddings = self._embed_texts([text], prefix="query: ")
        return embeddings[0].tolist()


def get_embeddings_model(
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 32
) -> E5Embeddings:
    """
    Factory function pour créer une instance du modèle d'embeddings E5.

    Args:
        model_id: Identifiant du modèle (par défaut depuis .env ou multilingual-e5-large)
        device: Device à utiliser (auto-détecté si None)
        batch_size: Taille des batchs pour le traitement

    Returns:
        E5Embeddings: Instance du modèle d'embeddings configuré
    """
    if model_id is None:
        model_id = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

    logger.info(f"Initialisation du modèle d'embeddings: {model_id}")
    return E5Embeddings(
        model_id=model_id,
        device=device,
        batch_size=batch_size
    )


def main():
    """
    Fonction de test pour vérifier le fonctionnement du modèle d'embeddings.
    """
    logger.info("="*70)
    logger.info("TEST DU MODULE EMBEDDINGS")
    logger.info("="*70)

    # Créer le modèle
    embeddings = get_embeddings_model()

    # Textes de test
    test_documents = [
        "Le Louvre abrite la Joconde.",
        "La Tour Eiffel est un monument emblématique de Paris.",
        "Le château de Versailles attire des millions de visiteurs chaque année."
    ]

    test_query = "Musée à Paris"

    # Test des embeddings de documents
    logger.info("\n1. Encodage de documents:")
    doc_embeddings = embeddings.embed_documents(test_documents)
    logger.info(f"   Nombre de documents: {len(doc_embeddings)}")
    logger.info(f"   Dimension des embeddings: {len(doc_embeddings[0])}")

    # Test de l'embedding de requête
    logger.info("\n2. Encodage d'une requête:")
    query_embedding = embeddings.embed_query(test_query)
    logger.info(f"   Requête: '{test_query}'")
    logger.info(f"   Dimension de l'embedding: {len(query_embedding)}")

    # Calcul de similarité (cosinus)
    logger.info("\n3. Similarités (cosinus):")
    query_vec = np.array(query_embedding)
    for i, doc in enumerate(test_documents):
        doc_vec = np.array(doc_embeddings[i])
        similarity = np.dot(query_vec, doc_vec)  # Les vecteurs sont déjà normalisés
        logger.info(f"   Doc {i+1}: {similarity:.4f} - {doc[:50]}...")

    logger.info("\n" + "="*70)
    logger.info("✓ TEST TERMINÉ AVEC SUCCÈS")
    logger.info("="*70)


if __name__ == "__main__":
    main()
