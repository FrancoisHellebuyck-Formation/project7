# Rapport technique – Assistant intelligent de recommandation d’événements culturels

## 1. Objectifs du projet
### Contexte : 

- Puls-Events est une entreprise technologique innovante spécialisée dans le développement d'une plateforme de recommandations culturelles personnalisées.

- Pour améliorer l'expérience utilisateur et répondre à l'évolution des attentes numériques, Puls-Events souhaite intégrer un assistant intelligent capable de gérer les requêtes des utilisateurs en temps réel.

### Problématique :

- Un système RAG (Retrieval-Augmented Generation) répond aux besoins métier de Puls-Events en résolvant la problématique centrale de l'accès précis et fiable à l'information événementielle par un chatbot.

- Le RAG surmonte les limites des modèles de langage classiques (LLMs) pour fournir une solution à la fois performante et économiquement viable.

### Objectif du POC :

- L'objectif du POC pour Puls-Events est de démontrer de manière concrète et mesurable que la technologie RAG (Retrieval-Augmented Generation), en utilisant LangChain, Mistral et FAISS, est la solution optimale pour alimenter le futur chatbot d'événements culturels.

- Cet objectif se décline en trois axes principaux : 
    - la faisabilité technique, 
    - la valeur métier 
    - et la performance.

1. Démontrer la Faisabilité Technique 🛠️

Il s'agit de prouver que l'intégration des composants clés est fonctionnelle et stable, menant à une solution prête pour l'industrialisation.

Intégration du Pipeline Complet : Prouver la capacité à orchestrer le flux de données de bout en bout : de l'extraction des données d'événements récents via l'API Open Agenda, à leur transformation en embeddings, leur stockage dans l'index FAISS, et leur utilisation par le LLM Mistral via LangChain pour la génération de la réponse.

Portabilité et Déploiement : Valider la capacité à livrer un système standardisé et reproductible grâce à la conteneurisation Docker et à l'exposition via une API REST (FastAPI).

2. Démontrer la Valeur Métier (Pertinence) ✨

L'objectif est de s'assurer que le système répond directement aux besoins de l'utilisateur final et de l'entreprise Puls-Events.

Véracité des Réponses : Démontrer que le RAG élimine les "hallucinations" en basant systématiquement les réponses sur le contexte factuel et à jour des événements (dates, lieux, artistes, genres) extrait d'Open Agenda. C'est la validation de la fiabilité de l'information.

Expérience Utilisateur Améliorée : Prouver que le chatbot peut gérer et répondre avec fluidité à une grande variété de questions en langage naturel, y compris les requêtes sémantiques complexes (e.g., "Je cherche quelque chose de familial le week-end prochain") basées sur le jeu de test annoté.

Efficacité Opérationnelle : Montrer que ce système est plus rentable et plus rapide à actualiser qu'une approche de fine-tuning du LLM, car seule la base vectorielle a besoin d'être mise à jour avec les nouveaux événements.

3. Démontrer la Performance (Qualité et Rapidité) ⚡

Il faut quantifier l'efficacité du système à la fois sur la recherche et la génération.

Performance du Retrieval : Mesurer l'efficacité de FAISS à remonter les fragments de texte pertinents. Le Hit Rate (pourcentage de fois où le bon fragment est dans les top-k résultats) est la métrique clé pour valider que la bonne information est trouvée.

Qualité de la Génération : Mesurer la fidélité (faithfulness) et la pertinence de la réponse générée par Mistral par rapport au contexte fourni. La réponse doit être bien rédigée, concise et répondre directement à la question de l'utilisateur.

Latence (Temps de Réponse) : S'assurer que le système complet (API + RAG) offre un temps de réponse acceptable pour une expérience utilisateur fluide (cible typique : quelques secondes ou moins).


- Périmètre : Zone géographique ciblée, période d’événements, données utilisées.

## Architecture du système
Schéma global (schéma UML) :
Données entrantes (API Open Agenda)
Prétraitement / embeddings / base vectorielle
Intégration LLM avec LangChain
Exposition via API
Technologies utilisées :


3. Préparation et vectorisation des données
Source de données : API Open Agenda (paramètres utilisés, filtres appliqués)
Nettoyage : Exemples d’anomalies corrigées, méthodes utilisées
Chunking : Raison du découpage, taille choisie
Embedding :
Modèle utilisé (ex. : Mistral embedding API)
Dimensionnalité, logique de batch, format des vecteurs

4. Choix du modèle NLP
Modèle sélectionné :
Pourquoi ce modèle ? (Critères : coût, qualité, compatibilité LangChain…)
Prompting (si utilisé) : Prompt de base / structure
Limites du modèle : 

5. Construction de la base vectorielle
Faiss utilisé :
Stratégie de persistance : 
Format de sauvegarde ?
nommage ?
Métadonnées associées : 
Ce qui est conservé pour chaque document

6. API et endpoints exposés
Framework utilisé : FastAPI / Flask
Endpoints clés :
/ask : question utilisateur, réponse du système
/rebuild : reconstruction de l’index (si besoin)
Format des requêtes/réponses
Exemple d’appel API : avec curl ou code Python
Tests effectués et documentés
Gestion des erreurs / limitations

7. Évaluation du système
Jeu de test annoté :
Nombre d’exemples
Méthode d’annotation
Métriques d’évaluation :
Exemple : similarité sémantique, taux de couverture des réponses, score de satisfaction (subjectif)
Résultats obtenus :
Analyse quantitative (scores globaux)
Analyse qualitative (exemples de bonnes/mauvaises réponses)
8. Recommandations et perspectives
Ce qui fonctionne bien
Limites du POC :
Volumétrie, performance, coût, couverture thématique ?
Améliorations possibles :
Ajout de…Amélioration de..
Passage en production via…

9.  Organisation du dépôt GitHub
Arborescence du dépôt (fichiers clés, scripts, dossiers)
Explication rapide de chaque répertoire

10. Annexes (exemples)
Extraits du jeu de test annoté
Prompt utilisé (si spécifique)
Extraits de logs ou exemples de réponse JSON
