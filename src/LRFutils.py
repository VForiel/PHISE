"""Petit module de secours pour LRFutils utilisé lors de la génération de docs.

Ce fichier fournit un symbole `color` minimal afin que la construction
de la documentation fonctionne même si la vraie dépendance n'est pas
disponible sur l'environnement CI.
"""
from typing import Any

def color(*args: Any, **kwargs: Any) -> str:
    """Retourne une couleur par défaut ou un identifiant simplifié.

    La fonction est volontairement minimaliste : elle retourne une couleur
    (string) utilisable dans matplotlib.
    """
    return 'tab:blue'
