# Intégration en temps réel des équations de la diffusion de Rayleigh dans une atmosphère

# Description du TIPE
Le but de ce TIPE est de créer un shader pouvant approcher la diffusion de la lumière dans une atmosphère, et capable de tourner en temps réel, à l'aide d'un modèle mathématique. Le code de ce TIPE est divisé en 2 parties.

# Parties du code du TIPE
## Utilisation du modèle
Le dossier `./AtmosphericShader/` contient un projet Godot 4.4. Il consiste en un modèle 3D de sphère (approché car composé de triangles) représentant la Terre, et le choix entre deux shaders:\
-"BASIC" fait le calcul en utilisant l'équation de la diffusion supposant un seul événement de diffusion\
-"FAST" utilise un petit réseau de neurones pour approcher la sortie de "BASIC", en espérant le faire plus rapidement\
Touches de déplacement:\
Les touches ZSQDAE permettent le mouvement de la caméra, la souris permet sa rotation.\
Shift divise la sensibilité par 20, et Ctrl par 400 (donc Shift+Ctrl par 8000), pour permettre des ajustements précis de la position de la caméra.\
La touche Shift divise aussi la sensibilité de la souris par 100.\
La touche M permet d'échanger entre "BASIC" et "FAST".\
La touche P permet de désactiver ou d'activer le texte montrant le temps par image et le choix de shader.

## Entraînement du modèle
Le dossier `./AtmosphericShader/approx/` contient le code nécessaire pour entraîner un modèle.\
Le fichier `simulation.c` permet de calculer la lumière reçue par l'oeil par rapport aux 4 coordonnées dont elle dépend. La symétrie par rapport à l'axe Terre-Soleil a permis de réduire ce nombre de coordonnées, valant initialement 5; cette réduction rend le calcul direct possible en un temps raisonnable, allant de quelques minutes à plusieurs jours, dépendant des paramètres NPOINTS, TAU_POINTS et I_POINTS choisis. Le temps de calcule évolue en O(NPOINTS^4 * TAU_POINTS * I_POINTS).\
Le fichier `to_tex.py` permet de convertir les données brutes générées par `simulation.c`, pour permettre à un humain de les visualiser.\
Le fichier `nn.py` permet d'entraîner un réseau de neurones sur des échantillons créés par `sampler.c`, et créer un modèle, qui peut ensuite être utilisé dans le projet Godot. Quelques exemples de modèles sont données dans `./params.gdshaderinc`, ou `./AtmosphericShader/runs` (moins bons).
