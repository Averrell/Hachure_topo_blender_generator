TOPO HACHURES 4.1 — Blender 5.x
================================

PRINCIPE
--------
Topo Hachures génère des hachures vectorielles à partir d'une heightmap. Les
traits suivent l'aspect local du terrain et sont contrôlés du bas vers le haut
par des courbes de niveau invisibles, selon la méthode QGIS de référence.

La 4.1 contient deux processus géométriques déterministes et indépendants,
auxquels peut s'ajouter une modulation continue par niveau :

1. HACHURES — PENTE
   La pente moyenne de chaque segment de contour règle l'espacement entre les
   valeurs minimale et maximale choisies. Cette couche reste strictement
   identique, que l'ombrage soit activé, désactivé ou orienté autrement.

2. HACHURES — OMBRAGE
   Un raster d'exposition absolu compare, pixel par pixel, la direction locale
   de descente à l'azimut choisi. Il déclenche un second passage qui génère
   uniquement des hachures supplémentaires sur les faces exposées.

Il n'existe aucune moyenne par carte, montagne ou contour, aucun tirage
aléatoire et aucune redistribution des hachures de pente. Les deux couches
utilisent le même traceur naturel, les mêmes champs d'aspect, les mêmes masques
et les mêmes règles de courbure.

DENSITÉ CONTINUE SELON LE NIVEAU
--------------------------------
L'option « Densité selon le niveau » module l'espacement déjà calculé sans
modifier la direction, la longueur ou la courbure naturelle des hachures :

- noir : hachures plus rares ;
- gris moyen : densité locale inchangée ;
- blanc : hachures plus denses ;
- toutes les nuances intermédiaires produisent une transition continue ;
- à 50 %, les multiplicateurs sont environ 0,5× / 1× / 1,5× ;
- l'intensité à 0 % ou l'option désactivée reproduit exactement la 4.0 ;
- aucun tirage aléatoire et aucune suppression après génération ne sont utilisés.

La modulation se combine avec la pente et l'ombrage directionnel. Elle ne
crée jamais de traits sous le seuil de pente minimale.

DENSITÉ QGIS SELON LA PENTE
---------------------------
- sous « Pente minimale » : aucune hachure ;
- pente faible retenue : espacement proche du maximum ;
- pente forte : espacement proche du minimum ;
- transition progressive entre les deux seuils de pente ;
- « Influence de la pente sur la densité » à 100 % reproduit la variation
  complète ; à 0 %, toutes les pentes retenues utilisent l'espacement maximal ;
- des espacements minimal et maximal identiques donnent une densité uniforme.

OMBRAGE DIRECTIONNEL INDÉPENDANT
--------------------------------
- 0° = nord, 90° = est, 180° = sud, 270° = ouest ;
- la rose des vents propose huit directions rapides ;
- l'intensité règle la quantité de hachures supplémentaires ;
- à 100 %, une face parfaitement alignée peut recevoir jusqu'à 100 % de
  départs supplémentaires par rapport à sa densité QGIS locale ;
- à 200 %, elle peut recevoir jusqu'à 200 % de départs supplémentaires, soit
  deux fois la couche d'ombre produite à 100 % ;
- l'effet diminue continûment avec l'écart angulaire ;
- à 90°, sur la face opposée ou sous la pente minimale : aucun ajout ;
- les nouveaux traits sont intercalés de manière déterministe, sans copier une
  courbe existante.

SVG ET CALQUES INKSCAPE
-----------------------
Le SVG contient deux vrais calques Inkscape :

- « Hachures — pente » ;
- « Hachures — ombrage » lorsque l'option est active.

Le calque d'ombrage peut être masqué sans affecter la couche de pente. Les
réglages sont intégrés aux métadonnées du SVG, sans chemin personnel.

RÉGLAGES PRINCIPAUX
-------------------
- Coupure tous les niveaux : intervalle vertical des contours de contrôle ;
- Longueur des segments : zone utilisée pour mesurer la pente moyenne ;
- Espacements min/max : densité QGIS des pentes fortes/faibles ;
- Influence de la pente : amplitude de cette variation ;
- Pentes min/max : seuil d'apparition et seuil de densité maximale ;
- Épaisseur : constante pour toutes les hachures ;
- Supprimer les micro-traits : filtre facultatif des fragments courts.

MASQUES ET BORDURES
-------------------
Jusqu'à trois masques PNG peuvent être fusionnés. Blanc opaque signifie
« exclure » ; noir ou transparent signifie « conserver ». Les masques doivent
avoir le même cadrage et les mêmes proportions que la heightmap.

« Exclure les bordures » détecte les différentes formes et îlots de la carte.
Une marge continue éloigne les traits des bâtiments, murets et autres zones
exclues. Les valeurs décimales (`0.1`, `0.25`, `0.5`, etc.) sont réellement
appliquées dans l'aperçu, la reconstruction du relief et l'export. La valeur
exacte fait partie de la signature du cache.

CACHE ET PERFORMANCES
---------------------
Les caches d'analyse du relief et de contours V18/2.0 restent réutilisables.
La géométrie 4.1 possède une signature séparée lorsque la densité par niveau
est active. Avec l'ombrage actif, un second
passage géométrique est nécessaire.

La gestion du cache affiche sa taille et son nombre de variantes. Le bouton de
suppression demande une confirmation.

INSTALLATION
------------
1. Désactiver l'ancienne version de Topo Hachures.
2. Redémarrer Blender.
3. Edit > Preferences > Add-ons > Install from Disk.
4. Choisir topo_hachures_4_1.zip.
5. Activer « Topo Hachures 4.1 SVG / PNG ».
6. Vue 3D > touche N > onglet « Topo Hachures ».

RÉFÉRENCES
----------
https://somethingaboutmaps.wordpress.com/2024/07/07/automated-hachuring-in-qgis/
https://github.com/pinakographos/Hachures
