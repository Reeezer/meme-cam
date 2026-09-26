# Meme-Cam 🦆

Un filtre de webcam : tu fais un geste, le mème qui fait **le même geste** se pose sur ton visage. En direct, et dans Google Meet, Zoom ou Teams.

Vu sur Instagram : [@askleonm](https://www.instagram.com/askleonm/)

## Les gestes

| Geste | Mème (fichier dans `memes/`) |
|---|---|
| 🙌 deux mains ouvertes levées, de part et d'autre de la tête | Absolute Cinema — `absolute-cinema.jpg` |
| 🙏 mains jointes devant la poitrine | SpongeBob « please » — `bob-please.jpg` |
| 🫶 cœur avec les mains (index joints en haut, pouces en bas) | Justin Bieber — `bieber-coeur.jpg` |
| ✋ une paume ouverte levée à côté du visage | Drake « non » — `drake-non.jpg` |
| 👉 bras tendu, index pointé loin de la tête | DiCaprio — `dicaprio-pointe.jpg` |
| 🤔 index sur la tempe | Roll Safe — `roll-safe.jpg` |
| 🤦 main ouverte sur le haut du visage | Jonah Hill — `jonah-hill.jpg` |

Tiens chaque geste environ une demi-seconde. Le lien entre gestes et images est dans [`memes/memes.json`](memes/memes.json) : change les images comme tu veux.

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et [ffmpeg](https://ffmpeg.org/) (pour l'enregistrement).

```bash
git clone https://github.com/Reeezer/meme-cam.git
cd meme-cam
uv sync
```

**Les images des mèmes ne sont pas fournies** (droits d'auteur) : télécharge-les (par exemple sur [imgflip](https://imgflip.com/memetemplates)) et mets-les dans `memes/` avec les noms du tableau. Les modèles MediaPipe (~15 Mo) se téléchargent tout seuls au premier lancement.

## Utilisation

```bash
uv run app.py                          # fenêtre d'aperçu (Q ou Échap pour quitter)
uv run app.py --hud                    # + ce que la caméra détecte : cadre du visage, points des mains
uv run app.py --enregistrer demo.mp4   # + enregistre le résultat
uv run app.py --taille 0.7             # hauteur des mèmes (part de la hauteur de l'image, 0,6 par défaut)
```

### En visio (Meet, Zoom, Teams)

1. Installe [OBS Studio](https://obsproject.com/) : il fournit la webcam virtuelle.
2. Lance `uv run app.py --visio`.
3. Dans ta visio, choisis la caméra **« OBS Virtual Camera »**.

## Régler les gestes sur tes mains

Les seuils sont dans [`gestes.py`](gestes.py), mesurés en largeurs de visage (ils ne dépendent donc pas de ta distance à la caméra).

```bash
uv run app.py --debug             # affiche les mesures en direct
uv run app.py --brut essais.mp4   # enregistre ta webcam pendant que tu fais les gestes…
uv run analyser.py essais.mp4     # …puis relis, image par image, ce que les règles mesurent
uv run app.py --source essais.mp4 --hud --enregistrer demo.mp4   # rejoue l'enregistrement dans l'appli
uv run pytest                     # tests des règles, sur des mains synthétiques
```

## Comment ça marche

- [MediaPipe](https://ai.google.dev/edge/mediapipe) trouve les mains (21 points chacune) et le visage (478 points). Loin de la caméra (au-delà d'environ 1,5 m), le modèle du visage ne voit plus rien : le modèle du corps prend le relais pour situer la tête.
- Chaque geste est une règle simple sur ces points (doigts tendus, distance de l'index à la tempe, écart des poignets…), mesurée en largeurs de visage.
- Un geste doit être vu plusieurs images de suite avant que son mème apparaisse : pas de clignotement quand les mains passent d'un geste à l'autre.

## English

Meme-Cam is a webcam filter: make a gesture and the meme doing the same gesture lands on your face, live, including in Google Meet, Zoom or Teams (through OBS Studio's virtual camera). Gesture detection uses MediaPipe hands, face and pose landmarks with simple, tested rules. Meme images are not included (copyright): drop your own in `memes/` using the file names above. Install with `uv sync`, run with `uv run app.py --hud`.

## Licence

[MIT](LICENSE) pour le code. Les images de mèmes restent la propriété de leurs auteurs.
