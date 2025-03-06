# Contenu proceeding et poster (SPIE & SF2A)

- Présentation de l'architecture
- Optimisation
  - [x] Algorithme d'optimisation
  - [ ] Preuve empirique : répartition des retards trouvés en fonction des erreurs de retard
  - [x] Limitation : même après optimisation, montrer que les cartes de transmission ne sont exploitable que dans le cas où les retards injectés sont parfaitement opposés aux erreurs de retards -> les combinaisons de retard "dégénérés" ne réduisent pas la détectabilté du compagnon mais induisent en erreur sur sa position
- Performance
  - [x] Test de ROC sur différents estimateus sur les kernels
  - [ ] Test de ROC sur différents estimateurs sur la sortie "gathered"
  - [ ] Avec la meilleur méthode de détection : carte de détectabilité avec probabilité de fausse alarme a 1%

# Conetnu 1er papier

Contenu précédent + :
- Diversité angulaire (rotation de ligne de base)
  - [ ] Extraction d'une fonction via la carte de transmission
  - [ ] Tentative de fit cette fonction avec les données récoltées
- Amélioration de a méthode d'optimisation
  - [ ] Etudier les retards au travers de modes plutôt que via des retardateurs indépendants (cf. https://github.com/Leirof/Tunable-Kernel-Nulling/discussions/31#discussioncomment-8669317)
  - [ ] Etudier d'autres métriques pouvent réduire la dégénérscence des retards injectés
- Résultats en labo
  - [ ] Etudier l'efficacité de l'algorithme d'optimisation
  - [ ] Comparer les performances finales à celles obtenues numériquement