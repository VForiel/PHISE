"""Gestion d'une caméra virtuelle pour la simulation d'interférométrie.

Ce module fournit la classe Camera qui représente un capteur simple utilisé
pour convertir des champs électriques complexes (visibilités) en un nombre
de photons détectés pendant une durée d'exposition. La classe est volontairement
minimaliste et conçue pour être utilisée par un objet `Interferometer` (type
indiqué dans les annotations) qui agit comme parent.

Principales responsabilités
- stocker le temps d'exposition (attribut `e`) sous la forme d'une
    quantité Astropy en secondes;
- simuler la détection de photons via la méthode `acquire_pixel` à partir
    d'un tableau de champs électriques complexes ;
- supporter un mode `ideal` (pas de bruit, valeur entière tronquée)
    ou un mode réaliste (bruit de Poisson / approximations gaussiennes pour
    grands nombres de photons).

Classe exposée
- Camera

Exemple minimal
---------------
>>> from phise.classes.camera import Camera
>>> import numpy as np
>>> import astropy.units as u
>>> cam = Camera(e=0.5 * u.s, ideal=False, name='CamSim')
>>> psi = np.array([1+0j, 0.5+0.5j])  # champs électriques complexes
>>> n = cam.acquire_pixel(psi)

Erreurs levées
- TypeError si `e` n'est pas une `astropy.units.Quantity` ou si `ideal`
    / `name` ne sont pas des types attendus.
- ValueError si `e` ne peut pas être converti en unité de temps.

Notes
- Les valeurs d'entrée et sortie sont minimales par conception :
    - `ψ` (psi) est attendu comme un numpy.ndarray de nombres complexes où
        |ψ|^2 donne un flux en s^-1 par élément ;
    - `acquire_pixel` retourne un entier correspondant au nombre de photons
        détectés pendant l'exposition `e`.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .interferometer import Interferometer
import numpy as np
import astropy.units as u
import numba as nb
import math

class Camera:
    """Représentation d'une caméra virtuelle utilisée pour la détection de photons.

    Cette classe encapsule des paramètres simples d'un capteur (temps
    d'exposition, nom, mode idéal) et fournit une méthode pour convertir
    des champs électriques complexes en nombre de photons détectés durant
    l'exposition.
    """

    __slots__ = ('_parent_interferometer', '_e', '_name', '_ideal')

    def __init__(self, e: Optional[u.Quantity] = None, ideal: bool = False, name: str = 'Unnamed Camera'):
        """Initialise la caméra.

        Paramètres
        ----------
        e : Optional[astropy.units.Quantity]
            Temps d'exposition. Doit être une quantité avec une unité de temps
            (par exemple ``1 * u.s``). Si ``None`` la valeur par défaut ``1 s``
            est utilisée lorsque possible.
        ideal : bool
            Si True, la caméra est considérée idéale et renverra la valeur
            entière attendue sans bruit (tronquée). Si False, un bruit de
            comptage (Poisson) est simulé.
        name : str
            Nom lisible pour la caméra.

        Exceptions
        ----------
        TypeError
            Si ``ideal`` n'est pas un booléen ou si ``name`` n'est pas une
            chaîne de caractères.
        ValueError
            Si la quantité ``e`` ne peut pas être convertie en unité de
            temps.
        """
        self._parent_interferometer = None
        # avoid evaluating `1 * u.s` at import time which may fail when astropy is mocked
        if e is None:
            try:
                e = 1 * u.s
            except Exception:
                e = None
        self.e = e
        self.ideal = ideal
        self.name = name

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2f}'
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def e(self) -> u.Quantity:
        """Temps d'exposition de la caméra en secondes.

        Retourne une ``astropy.units.Quantity`` exprimée en secondes. L'accesseur
        n'effectue pas de conversion supplémentaire ; la conversion est gérée
        dans le setter.
        """
        return self._e

    @e.setter
    def e(self, e: u.Quantity):
        """Définit le temps d'exposition.

        Paramètres
        ----------
        e : astropy.units.Quantity
            Quantité représentant un temps (ex : ``0.5 * u.s``).

        Exceptions
        ----------
        TypeError
            Si ``e`` n'est pas un ``astropy.units.Quantity``.
        ValueError
            Si la quantité ne peut pas être convertie en une unité de temps.
        """
        if not isinstance(e, u.Quantity):
            raise TypeError('e must be an astropy Quantity')
        try:
            e = e.to(u.s)
        except u.UnitConversionError:
            raise ValueError('e must be in a time unit')
        self._e = e

    @property
    def parent_interferometer(self) -> Interferometer:
        """Référence en lecture seule vers l'interféromètre parent.

        Le setter est en lecture seule et l'attribut doit être configuré par
        l'objet parent (ex. une instance d'``Interferometer``) si nécessaire.
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, _):
        """Tentative d'écriture interdite pour `parent_interferometer`.

        Cette propriété est en lecture seule ; toute affectation directe
        déclenche une exception.
        """
        raise ValueError('parent_interferometer is read-only')

    @property
    def ideal(self) -> bool:
        """Indique si la caméra est en mode idéal (sans bruit).

        Retourne ``True`` lorsque le bruit de détection est désactivé. Le mode
        idéal est utile pour des tests déterministes.
        """
        return self._ideal

    @ideal.setter
    def ideal(self, ideal: bool):
        """Définit le mode idéal de la caméra.

        Paramètres
        ----------
        ideal : bool
            ``True`` pour un capteur sans bruit, ``False`` pour simuler le
            bruit de comptage.

        Exceptions
        ----------
        TypeError
            Si ``ideal`` n'est pas un booléen.
        """
        if not isinstance(ideal, bool):
            raise TypeError('ideal must be a boolean')
        self._ideal = ideal

    @property
    def name(self) -> str:
        """Nom lisible de la caméra.

        Retourne une chaîne de caractères représentant le nom de l'objet.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Définit le nom de la caméra.

        Paramètres
        ----------
        name : str
            Nom lisible. Une ``TypeError`` est levée si ``name`` n'est pas une
            chaîne.
        """
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    def acquire_pixel(self, ψ: np.ndarray[complex]) -> int:
        """Simule l'acquisition d'un pixel à partir de champs électriques.

        La méthode calcule le nombre moyen de photons attendus comme la somme
        des puissances |ψ|^2 multipliée par le temps d'exposition ``e``. Puis
        elle simule la détection selon le mode :

        - si ``ideal`` : retourne la valeur entière tronquée de l'espérance
          (déterministe) ;
        - sinon : pour des espérances raisonnables (<= 2e9) on tire d'une loi
          de Poisson ; pour des espérances très grandes on utilise une
          approximation gaussienne pour éviter des problèmes de performance.

        Paramètres
        ----------
        ψ : numpy.ndarray de nombres complexes
            Tableau 1D (ou broadcastable) contenant les amplitudes complexes
            du champ électrique (unités : s**(-1/2)).

        Retour
        -----
        int
            Nombre de photons détectés pendant l'exposition.

        Remarques
        ---------
        - La méthode retourne un entier >= 0. Les détails numériques (seuil
          2e9) sont choisis empiriquement pour basculer vers une approximation
          normale lorsque la loi de Poisson devient coûteuse.
        - Pour des usages reproductibles, régler le germe du générateur aléatoire
          avant d'appeler la méthode (par ex. via ``np.random.seed(...)``).
        """
        expected_photons = np.sum(np.abs(ψ) ** 2) * self.e.to(u.s).value
        if self.ideal:
            detected_photons = int(expected_photons)
        elif expected_photons <= 2000000000.0:
            detected_photons = np.random.poisson(expected_photons)
        else:
            detected_photons = int(expected_photons + np.random.normal(0, math.sqrt(expected_photons)))
        return detected_photons