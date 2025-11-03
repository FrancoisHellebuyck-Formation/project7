# 🎨 Interface Utilisateur Streamlit

Ce package contient l'interface utilisateur web pour le chatbot Puls-Events.

## 📋 Contenu

- `chatbot.py` - Application Streamlit principale avec historique de conversation

## 🚀 Utilisation

### Démarrage rapide

```bash
# 1. Démarrer l'API RAG (dans un terminal)
make run-api

# 2. Démarrer l'interface Streamlit (dans un autre terminal)
make run-ui
```

L'interface sera accessible sur **http://localhost:8501**

### Accès direct

```bash
uv run streamlit run src/ui/chatbot.py
```

## ✨ Fonctionnalités

### Interface principale

- 💬 **Chat interactif** : Interface conversationnelle intuitive
- 📝 **Historique complet** : Maintien de toutes les conversations
- 🎭 **Avatar Puls-Events** : Guide culturel personnalisé
- ⚙️ **Paramètres configurables** : Ajustement du nombre de documents contextuels

### Sidebar

- 🔗 **Statut API** : Indicateur de connexion en temps réel
- 📊 **Statistiques** : Nombre de messages et questions
- 🗑️ **Nouvelle conversation** : Réinitialisation de l'historique
- ℹ️ **Informations** : À propos du chatbot

### Détails des réponses

Pour chaque réponse, l'interface affiche :
- 🎯 **Tokens utilisés** : Coût de la requête
- 📚 **Documents contextuels** : Nombre de sources utilisées
- 🔍 **Sources** : Liste des événements trouvés avec scores de pertinence

## 🎯 Architecture

```
Utilisateur → Streamlit UI → API /ask → RAG + Mistral AI → Réponse
                ↓
         Historique session
```

### Gestion de l'état

L'application utilise `st.session_state` pour :
- `messages` : Liste de tous les messages (user + assistant)
- `conversation_started` : Indicateur de première utilisation

### Format des messages

```python
{
    "role": "user" | "assistant",
    "content": "Message texte",
    "timestamp": "ISO 8601",
    "metadata": {
        "tokens_used": {...},
        "context_count": int,
        "sources": str
    }
}
```

## 🔧 Configuration

Variables d'environnement (`.env`) :

```bash
RAG_API_URL=http://localhost:8000  # URL de l'API
RAG_TOP_K=5                         # Nombre de documents par défaut
```

## 🎨 Personnalisation

### Modifier les avatars

Dans `chatbot.py`, ligne ~145 :
```python
avatar = "🧑" if role == "user" else "🎭"
```

### Modifier le message de bienvenue

Dans `chatbot.py`, fonction `main()`, section "Message de bienvenue".

### Ajuster les paramètres

Le slider dans la sidebar permet d'ajuster dynamiquement le nombre de documents contextuels (k) entre 1 et 10.

## 📊 Fonctionnalités avancées

### Gestion d'erreurs

- ❌ **Connexion API** : Message d'erreur si l'API est inaccessible
- ⏱️ **Timeout** : Délai de 30 secondes pour les requêtes
- 🔴 **Statut** : Indicateur visuel de l'état de l'API

### Performance

- 🚀 **Réponses instantanées** : Affichage progressif avec spinner
- 💾 **Historique léger** : Stockage en session uniquement
- 🔄 **Rechargement** : `st.rerun()` pour rafraîchir l'interface

## 🐛 Dépannage

### L'API est déconnectée

```bash
# Vérifier que l'API tourne
curl http://localhost:8000/health

# Redémarrer l'API si nécessaire
make run-api
```

### Streamlit ne se lance pas

```bash
# Vérifier que streamlit est installé
uv run streamlit --version

# Réinstaller si nécessaire
uv add streamlit
```

### Port 8501 déjà utilisé

```bash
# Tuer le processus existant
lsof -ti:8501 | xargs kill -9

# Ou utiliser un autre port
uv run streamlit run src/ui/chatbot.py --server.port 8502
```

## 📚 Documentation Streamlit

- [Documentation officielle](https://docs.streamlit.io/)
- [Chat elements](https://docs.streamlit.io/library/api-reference/chat)
- [Session state](https://docs.streamlit.io/library/api-reference/session-state)
