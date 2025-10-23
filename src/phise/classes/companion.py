"""Représentation d'un compagnon ponctuel dans le plan du ciel.

Ce module définit la classe `Companion` qui modélise une source ponctuelle
(non résolue) entourant une étoile hôte. Un compagnon est caractérisé par
sa *contraste* par rapport à l'étoile, sa séparation angulaire et son
angle parallactique.

La classe effectue des validations simples sur les unités (utilise
``astropy.units``) et expose des propriétés en lecture/écriture pour les
paramètres physiques. La relation vers l'objet parent (`Target`) est
en lecture seule.

Exemple
-------
>>> from phise.classes.companion import Companion
>>> import astropy.units as u
>>> comp = Companion(c=1e-3, θ=100*u.mas, α=0.1*u.rad, name='b')
>>> print(comp)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from target import Target
from astropy import units as u

class Companion:
    """Source ponctuelle représentant un compagnon astronomique.

    Paramètres principaux
    - c : contraste (float, positif) par rapport à l'étoile hôte
    - θ : séparation angulaire (astropy.Quantity, ex : milliarcsecondes)
    - α : angle parallactique (astropy.Quantity, ex : radians)
    - name : nom lisible (str)

    La classe valide les unités de ``θ`` et ``α`` lors de l'affectation
    et convertit respectivement ``θ`` en milliarcsecondes (mas) et ``α`` en
    radians pour un stockage interne cohérent.
    """

    __slots__ = ('_parent_target', '_c', '_θ', '_α', '_name')

    def __init__(self, c: float, θ: u.Quantity, α: u.Quantity, name: str = 'Unnamed Companion'):
        """Initialise un compagnon ponctuel.

        Paramètres
        ----------
        c : float
            Contraste du compagnon par rapport à l'étoile hôte (doit être >= 0).
        θ : astropy.units.Quantity
            Séparation angulaire (par ex. ``100 * u.mas``).
        α : astropy.units.Quantity
            Angle parallactique (par ex. ``0.1 * u.rad``).
        name : str, optionnel
            Nom lisible du compagnon.

        Exceptions
        ----------
        TypeError
            Si les types fournis ne correspondent pas aux attentes.
        ValueError
            Si des valeurs physiques invalides sont fournies (p.ex. contraste négatif).
        """
        self._parent_target = None
        self.θ = θ
        self.α = α
        self.c = c
        self.name = name

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = f'Companion "{self.name}"\n'
        res += f'  Contrast: {self.c:.2f}\n'
        res += f'  Angular separation: {self.θ:.2f}\n'
        res += f'  Parallactic angle: {self.α:.2f}'
        return res

    @property
    def c(self) -> float:
        """Contraste du compagnon (nombre sans dimension).

        Retourne la valeur flottante du contraste, toujours positive.
        """
        return self._c

    @c.setter
    def c(self, c: float):
        """Définit le contraste du compagnon.

        Paramètres
        ----------
        c : float
            Contraste (>= 0). Une ``TypeError`` est levée si ``c`` n'est ni int ni float.
        """
        if not isinstance(c, (int, float)):
            raise TypeError('c must be a float')
        if c < 0:
            raise ValueError('c must be positive')
        self._c = float(c)

    @property
    def θ(self) -> u.Quantity:
        """Séparation angulaire du compagnon (astropy.Quantity en mas).

        La valeur renvoyée est une quantité en milliarcsecondes (mas).
        """
        return self._θ

    @θ.setter
    def θ(self, θ: u.Quantity):
        """Définit la séparation angulaire.

        Paramètres
        ----------
        θ : astropy.units.Quantity
            Quantité d'angle (ex : ``100 * u.mas`` ou ``0.1 * u.arcsec``).
        """
        if not isinstance(θ, u.Quantity):
            raise TypeError('θ must be an astropy Quantity')
        try:
            θ = θ.to(u.mas)
        except u.UnitConversionError:
            raise ValueError('θ must be an angle')
        self._θ = θ

    @property
    def α(self) -> u.Quantity:
        """Angle parallactique du compagnon (astropy.Quantity en radians).

        La valeur renvoyée est une quantité en radians.
        """
        return self._α

    @α.setter
    def α(self, α: u.Quantity):
        """Définit l'angle parallactique.

        Paramètres
        ----------
        α : astropy.units.Quantity
            Quantité d'angle (par ex. ``0.1 * u.rad`` ou ``10 * u.deg``).
        """
        if not isinstance(α, u.Quantity):
            raise TypeError('α must be an astropy Quantity')
        try:
            α = α.to(u.rad)
        except u.UnitConversionError:
            raise ValueError('α must be an angle')
        self._α = α

    @property
    def parent_target(self) -> Target:
        """Référence en lecture seule vers l'objet `Target` parent.

        Toute tentative d'affectation directe déclenche une erreur ; la
        relation doit être établie par l'objet parent.
        """
        return self._parent_target

    @parent_target.setter
    def parent_target(self, target: Target):
        """Propriété en écriture désactivée pour `parent_target`.

        L'accès en écriture provoque une exception pour signaler que
        l'association doit être gérée par l'objet parent.
        """
        raise ValueError('parent_target is read-only')

    @property
    def name(self) -> str:
        """Nom lisible du compagnon.

        Retourne une chaîne de caractères.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Définit le nom du compagnon.

        Paramètres
        ----------
        name : str
            Nom lisible. ``TypeError`` si non-str.
        """
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name