#!/usr/bin/env python3
"""
Point d'entrée principal pour l'exécution des pipelines.

Ce script valide l'environnement et exécute soit le pipeline statistique
soit le pipeline expérientiel, selon l'argument fourni.

Usage:
    python run.py statistical    # Exécute le pipeline statistique
    python run.py experiential   # Exécute le pipeline expérientiel
    python run.py                # Exécute le pipeline statistique par défaut
"""

import sys
import argparse
import logging
from pathlib import Path

# Configuration du logging avant tout import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def validate_environment_statistical():
    """
    Valide que l'environnement est correctement configuré pour le pipeline statistique.
    
    Returns
    -------
    bool
        True si l'environnement est valide, False sinon.
    """
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        LEXICON_DIR,
        CORPUS_PATH,
        PERCEPTUAL_LEXICON_PATH,
        MUSICO_TECH_LEXICON_PATH,
        PIPELINE_STATISTICAL_DIR
    )
    
    errors = []
    
    # Vérifier que nous sommes dans le bon répertoire
    if not PROJECT_ROOT.exists():
        errors.append(f"Racine du projet introuvable : {PROJECT_ROOT}")
    
    # Vérifier les répertoires principaux
    if not DATA_DIR.exists():
        errors.append(f"Répertoire de données introuvable : {DATA_DIR}")
    
    if not LEXICON_DIR.exists():
        errors.append(f"Répertoire de lexiques introuvable : {LEXICON_DIR}")
    
    # Vérifier les fichiers essentiels
    if not CORPUS_PATH.exists():
        errors.append(f"Corpus introuvable : {CORPUS_PATH}")
    
    if not PERCEPTUAL_LEXICON_PATH.exists():
        errors.append(f"Lexique perceptuel introuvable : {PERCEPTUAL_LEXICON_PATH}")
    
    if not MUSICO_TECH_LEXICON_PATH.exists():
        errors.append(f"Lexique musico-technique introuvable : {MUSICO_TECH_LEXICON_PATH}")
    
    # Vérifier que le pipeline statistique existe
    main_py = PIPELINE_STATISTICAL_DIR / "main.py"
    if not main_py.exists():
        errors.append(f"Fichier main.py du pipeline statistique introuvable : {main_py}")
    
    if errors:
        logger.error("Erreurs de validation de l'environnement (pipeline statistique) :")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Validation de l'environnement (pipeline statistique) réussie")
    return True


def validate_environment_experiential():
    """
    Valide que l'environnement est correctement configuré pour le pipeline expérientiel.
    
    Returns
    -------
    bool
        True si l'environnement est valide, False sinon.
    """
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        CORPUS_PATH,
        PIPELINE_EXPERIENTIAL_DIR
    )
    
    errors = []
    
    # Vérifier que nous sommes dans le bon répertoire
    if not PROJECT_ROOT.exists():
        errors.append(f"Racine du projet introuvable : {PROJECT_ROOT}")
    
    # Vérifier les répertoires principaux
    if not DATA_DIR.exists():
        errors.append(f"Répertoire de données introuvable : {DATA_DIR}")
    
    # Vérifier les fichiers essentiels
    if not CORPUS_PATH.exists():
        errors.append(f"Corpus introuvable : {CORPUS_PATH}")
    
    # Vérifier que le pipeline expérientiel existe
    gliner_py = PIPELINE_EXPERIENTIAL_DIR / "gliner_extraction.py"
    if not gliner_py.exists():
        errors.append(f"Fichier gliner_extraction.py introuvable : {gliner_py}")
    
    if errors:
        logger.error("Erreurs de validation de l'environnement (pipeline expérientiel) :")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Validation de l'environnement (pipeline expérientiel) réussie")
    return True


def check_dependencies_statistical():
    """
    Vérifie que les dépendances essentielles pour le pipeline statistique sont disponibles.
    
    Returns
    -------
    bool
        True si les dépendances sont disponibles, False sinon.
    """
    required_modules = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'sklearn'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Modules manquants pour le pipeline statistique : {', '.join(missing)}")
        logger.error("Installez-les avec : pip install -r requirements.txt")
        return False
    
    logger.info("Toutes les dépendances (pipeline statistique) sont disponibles")
    return True


def check_dependencies_experiential():
    """
    Vérifie que les dépendances essentielles pour le pipeline expérientiel sont disponibles.
    
    Returns
    -------
    bool
        True si les dépendances sont disponibles, False sinon.
    """
    required_modules = [
        'gliner',
        'pandas'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Modules manquants pour le pipeline expérientiel : {', '.join(missing)}")
        logger.error("Installez-les avec : pip install gliner pandas")
        return False
    
    logger.info("Toutes les dépendances (pipeline expérientiel) sont disponibles")
    return True


def run_statistical_pipeline():
    """
    Exécute le pipeline statistique.
    
    Returns
    -------
    int
        Code de sortie (0 pour succès, 1 pour erreur).
    """
    from config import PIPELINE_STATISTICAL_DIR
    
    main_py = PIPELINE_STATISTICAL_DIR / "main.py"
    
    if not main_py.exists():
        logger.error(f"Fichier main.py introuvable : {main_py}")
        return 1
    
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE STATISTIQUE")
    logger.info("=" * 60)
    
    try:
        # Ajouter le répertoire racine au path Python pour les imports
        from config import PROJECT_ROOT
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        # Exécuter main.py du pipeline statistique
        import runpy
        runpy.run_path(str(main_py), run_name="__main__")
        
        logger.info("=" * 60)
        logger.info("PIPELINE STATISTIQUE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("ERREUR LORS DE L'EXÉCUTION DU PIPELINE STATISTIQUE")
        logger.error("=" * 60)
        logger.exception(e)
        return 1


def run_experiential_pipeline():
    """
    Exécute le pipeline expérientiel (GLiNER).
    
    Returns
    -------
    int
        Code de sortie (0 pour succès, 1 pour erreur).
    """
    from config import PIPELINE_EXPERIENTIAL_DIR
    
    gliner_py = PIPELINE_EXPERIENTIAL_DIR / "gliner_extraction.py"
    
    if not gliner_py.exists():
        logger.error(f"Fichier gliner_extraction.py introuvable : {gliner_py}")
        return 1
    
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE EXPÉRIENTIEL (GLiNER)")
    logger.info("=" * 60)
    
    try:
        # Ajouter le répertoire racine au path Python pour les imports
        from config import PROJECT_ROOT
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        # Exécuter gliner_extraction.py
        import runpy
        runpy.run_path(str(gliner_py), run_name="__main__")
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXPÉRIENTIEL TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("ERREUR LORS DE L'EXÉCUTION DU PIPELINE EXPÉRIENTIEL")
        logger.error("=" * 60)
        logger.exception(e)
        return 1


def main():
    """
    Fonction principale d'exécution.
    """
    parser = argparse.ArgumentParser(
        description="Exécute les pipelines d'analyse de textes d'expériences d'écoute musicale"
    )
    parser.add_argument(
        'pipeline',
        nargs='?',
        default='statistical',
        choices=['statistical', 'experiential'],
        help='Pipeline à exécuter (statistical ou experiential). Par défaut: statistical'
    )
    
    args = parser.parse_args()
    
    if args.pipeline == 'statistical':
        logger.info("Vérification de l'environnement pour le pipeline statistique...")
        
        if not validate_environment_statistical():
            logger.error("L'environnement n'est pas correctement configuré pour le pipeline statistique.")
            return 1
        
        if not check_dependencies_statistical():
            logger.error("Des dépendances sont manquantes pour le pipeline statistique.")
            return 1
        
        return run_statistical_pipeline()
    
    elif args.pipeline == 'experiential':
        logger.info("Vérification de l'environnement pour le pipeline expérientiel...")
        
        if not validate_environment_experiential():
            logger.error("L'environnement n'est pas correctement configuré pour le pipeline expérientiel.")
            return 1
        
        if not check_dependencies_experiential():
            logger.error("Des dépendances sont manquantes pour le pipeline expérientiel.")
            return 1
        
        return run_experiential_pipeline()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
