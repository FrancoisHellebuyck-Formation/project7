"""
Configuration pytest pour les tests.

Ce fichier configure l'environnement de test, notamment le PYTHONPATH
pour permettre l'import des modules depuis src/.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src/ au PYTHONPATH
tests_dir = Path(__file__).parent
project_root = tests_dir.parent
src_dir = project_root / "src"

sys.path.insert(0, str(src_dir))
