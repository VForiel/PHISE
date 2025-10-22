Cette documentation est générée avec Sphinx.

Pour construire localement :

```powershell
python -m pip install -r docs/sphinx/requirements.txt
python -m sphinx -b html docs/sphinx docs/sphinx/_build/html
```

Fichiers importants :
- `docs/sphinx/conf.py` : configuration Sphinx
- `docs/sphinx/index.rst` : page d'accueil
- `docs/sphinx/modules.rst` : entrée autodoc pour le package `src`
