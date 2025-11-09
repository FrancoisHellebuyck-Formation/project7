# Makefile pour le projet OpenClassrooms Project 7
# Pipeline de traitement des données d'événements culturels

.PHONY: help install run-chunks run-embeddings run-vectorstore serve-vectorstore run-api run-agendas run-events clean lint format test docker-up docker-down

# Variables
PYTHON := python3
UV := uv
SRC_DIR := src

# Environment variables are loaded by Python scripts using python-dotenv
# No need to export from Makefile - this avoids parsing issues with special characters

# Couleurs pour l'affichage
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Affiche l'aide
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  OpenClassrooms Project 7 - Pipeline de données$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Installe les dépendances avec uv
	@echo "$(GREEN)📦 Installation des dépendances...$(NC)"
	$(UV) sync
	@echo "$(GREEN)✓ Dépendances installées$(NC)"

run-chunks: ## Lance le pipeline de chunking des documents
	@echo "$(GREEN)🔄 Lancement du pipeline de chunking...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/chunks/chunks_document.py
	@echo "$(GREEN)✓ Pipeline de chunking terminé$(NC)"

run-embeddings: ## Génère les embeddings et crée l'index FAISS (mode: recreate)
	@echo "$(GREEN)🧠 Génération des embeddings et création de l'index FAISS (RECREATE)...$(NC)"
	KMP_DUPLICATE_LIB_OK=TRUE $(UV) run $(PYTHON) $(SRC_DIR)/pipeline.py recreate
	@echo "$(GREEN)✓ Embeddings générés et index créé$(NC)"

run-update: ## Met à jour tout le pipeline (agendas → events → chunks → embeddings) en mode incrémental
	@echo "$(YELLOW)🔄 Mise à jour incrémentale complète du pipeline (UPDATE)...$(NC)"
	KMP_DUPLICATE_LIB_OK=TRUE $(UV) run $(PYTHON) $(SRC_DIR)/update_pipeline.py
	@echo "$(GREEN)✓ Pipeline mis à jour$(NC)"

show-last-update: ## Affiche les paramètres de la dernière exécution du pipeline
	@echo "$(BLUE)📊 Affichage des derniers paramètres utilisés...$(NC)"
	@$(UV) run $(PYTHON) $(SRC_DIR)/utils/show_last_update.py

show-history: ## Affiche l'historique des dernières exécutions (par défaut: 5)
	@echo "$(BLUE)📜 Affichage de l'historique des exécutions...$(NC)"
	@$(UV) run $(PYTHON) $(SRC_DIR)/utils/show_last_update.py --history 5

run-vectorstore: ## Démarre et teste le vector store existant
	@echo "$(GREEN)🔍 Démarrage du vector store...$(NC)"
	KMP_DUPLICATE_LIB_OK=TRUE $(UV) run $(PYTHON) $(SRC_DIR)/vectors/vectors.py
	@echo "$(GREEN)✓ Vector store testé$(NC)"

serve-vectorstore: ## Démarre le serveur de recherche vectorielle (mode interactif)
	@echo "$(GREEN)🚀 Démarrage du serveur de recherche vectorielle...$(NC)"
	KMP_DUPLICATE_LIB_OK=TRUE $(UV) run $(PYTHON) $(SRC_DIR)/vectors/server.py

run-api: ## Démarre l'API FastAPI de recherche
	@echo "$(GREEN)🌐 Démarrage de l'API FastAPI...$(NC)"
	@echo "$(YELLOW)   API disponible sur http://localhost:8000$(NC)"
	@echo "$(YELLOW)   Documentation sur http://localhost:8000/docs$(NC)"
	cd $(SRC_DIR) && KMP_DUPLICATE_LIB_OK=TRUE $(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-chat: ## Lance le chatbot Mistral CLI avec RAG
	@echo "$(GREEN)💬 Démarrage du chatbot Mistral CLI avec RAG...$(NC)"
	@echo "$(YELLOW)   Assurez-vous que l'API RAG est démarrée (make run-api)$(NC)"
	KMP_DUPLICATE_LIB_OK=TRUE $(UV) run $(PYTHON) $(SRC_DIR)/chat/mistral.py

cleanup-mongodb: ## Archive les collections MongoDB existantes (backup avec date)
	@echo "$(YELLOW)🗄️  Archivage des collections MongoDB...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/corpus/cleanup_mongodb.py
	@echo "$(GREEN)✓ Archivage terminé$(NC)"

run-agendas: ## Récupère les agendas depuis l'API OpenAgenda
	@echo "$(GREEN)📅 Récupération des agendas...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/corpus/get_corpus_agendas.py
	@echo "$(GREEN)✓ Agendas récupérés$(NC)"

run-events: ## Récupère les événements depuis l'API OpenAgenda
	@echo "$(GREEN)🎭 Récupération des événements...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/corpus/get_corpus_events.py
	@echo "$(GREEN)✓ Événements récupérés$(NC)"

deduplicate-events: ## Dédoublonne la collection MongoDB events (basé sur uid)
	@echo "$(GREEN)🔄 Dédoublonnement de la collection events...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/corpus/deduplicate_events.py
	@echo "$(GREEN)✓ Dédoublonnement terminé$(NC)"

clean-events: ## Nettoie les événements avec description insuffisante (<100 caractères)
	@echo "$(GREEN)🧹 Nettoyage des événements avec description courte...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/corpus/clean_events.py
	@echo "$(GREEN)✓ Nettoyage des événements terminé$(NC)"

run-all: cleanup-mongodb run-agendas run-events deduplicate-events clean-events run-chunks run-embeddings ## Lance le pipeline complet (cleanup → agendas → événements → dédoublonnement → nettoyage → chunks → embeddings)
	@echo "$(GREEN)✓ Pipeline complet terminé avec succès !$(NC)"

lint: ## Vérifie le code avec flake8
	@echo "$(YELLOW)🔍 Vérification du code avec flake8...$(NC)"
	$(UV) run flake8 $(SRC_DIR)
	@echo "$(GREEN)✓ Code vérifié$(NC)"

format: ## Formate le code (à implémenter avec black)
	@echo "$(YELLOW)✨ Formatage du code...$(NC)"
	@echo "$(RED)⚠️  Black non configuré. Ajoutez-le au pyproject.toml$(NC)"

test: ## Lance les tests unitaires avec pytest
	@echo "$(YELLOW)🧪 Lancement des tests unitaires...$(NC)"
	$(UV) run pytest tests/ -v
	@echo "$(GREEN)✓ Tests terminés$(NC)"

test-cov: ## Lance les tests avec rapport de couverture
	@echo "$(YELLOW)🧪 Lancement des tests avec couverture...$(NC)"
	$(UV) run pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)✓ Tests et couverture terminés$(NC)"

collect-ragas: ## Collecte les données (answer/contexts) via l'API /ask
	@echo "$(BLUE)📦 Collecte des données RAGAS via /ask...$(NC)"
	@$(UV) run python tests/collect_ragas_data.py
	@echo "$(GREEN)✓ Données collectées dans ragas_test_questions_collected.json$(NC)"

test-ragas: ## Lance l'évaluation RAGAS du système RAG
	@$(UV) run python tests/evaluate_ragas.py

docker-up: ## Démarre MongoDB avec Docker Compose
	@echo "$(GREEN)🐳 Démarrage de MongoDB...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ MongoDB démarré$(NC)"

docker-down: ## Arrête MongoDB
	@echo "$(YELLOW)🛑 Arrêt de MongoDB...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ MongoDB arrêté$(NC)"

docker-logs: ## Affiche les logs de MongoDB
	docker-compose logs -f

clean: ## Nettoie les fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage des fichiers temporaires...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Nettoyage terminé$(NC)"

clean-backups: ## Supprime les collections backup MongoDB (interactif)
	@echo "$(YELLOW)🧹 Nettoyage des collections backup MongoDB...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/utils/clean_backups.py

clean-backups-dry-run: ## Affiche les collections backup sans les supprimer
	@echo "$(BLUE)🔍 Liste des collections backup MongoDB...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/utils/clean_backups.py --dry-run

clean-backups-force: ## Supprime les collections backup MongoDB sans confirmation
	@echo "$(RED)⚠️  Suppression forcée des collections backup MongoDB...$(NC)"
	$(UV) run $(PYTHON) $(SRC_DIR)/utils/clean_backups.py --force

env-check: ## Vérifie que les variables d'environnement sont configurées
	@echo "$(BLUE)🔐 Vérification des variables d'environnement...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(RED)❌ Fichier .env non trouvé !$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Fichier .env trouvé$(NC)"
	@grep -q "MONGODB_URI" .env && echo "$(GREEN)✓ MONGODB_URI configuré$(NC)" || echo "$(RED)❌ MONGODB_URI manquant$(NC)"
	@grep -q "OA_API_KEY" .env && echo "$(GREEN)✓ OA_API_KEY configuré$(NC)" || echo "$(RED)❌ OA_API_KEY manquant$(NC)"

status: ## Affiche le statut du projet
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Statut du Projet$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)📁 Répertoire:$(NC) $(shell pwd)"
	@echo "$(YELLOW)🐍 Python:$(NC) $(shell $(PYTHON) --version 2>&1)"
	@echo "$(YELLOW)📦 UV:$(NC) $(shell $(UV) --version 2>&1 || echo 'Non installé')"
	@echo "$(YELLOW)🐳 Docker:$(NC) $(shell docker --version 2>&1 || echo 'Non installé')"
	@echo ""
	@$(MAKE) env-check
	@echo ""

# Alias pratiques
chunks: run-chunks ## Alias pour run-chunks
embeddings: run-embeddings ## Alias pour run-embeddings (mode recreate)
update: run-update ## Alias pour run-update (mise à jour incrémentale complète)
vectorstore: run-vectorstore ## Alias pour run-vectorstore
serve: serve-vectorstore ## Alias pour serve-vectorstore
api: run-api ## Alias pour run-api
agendas: run-agendas ## Alias pour run-agendas
events: run-events ## Alias pour run-events
deduplicate: deduplicate-events ## Alias pour deduplicate-events
cleanup: cleanup-mongodb ## Alias pour cleanup-mongodb
last-update: show-last-update ## Alias pour show-last-update
history: show-history ## Alias pour show-history
all: run-all ## Alias pour run-all
