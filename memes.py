"""Meme images and how they land on the face."""
import json
from pathlib import Path

import cv2
import numpy as np

DOSSIER = Path(__file__).parent / "memes"
POP = 0.15  # seconds of the pop-in (scale 0.8 -> 1)


def charger(config=DOSSIER / "memes.json"):
    """{gesture: BGR image}. memes.json maps each gesture to an image file in memes/;
    images are not versioned (copyrighted stills): everyone drops their own."""
    table = json.loads(Path(config).read_text(encoding="utf-8"))
    images, manquants = {}, []
    for geste, fichier in table.items():
        img = cv2.imread(str(DOSSIER / fichier), cv2.IMREAD_COLOR)
        if img is None:
            manquants.append(f"{geste} → memes/{fichier}")
        else:
            images[geste] = img
    return images, manquants


def boite_visage(points, largeur, hauteur, marge=1.9):
    """Square box (x, y, side) centred on the face, `marge` times the face size."""
    xs = [p.x * largeur for p in points]
    ys = [p.y * hauteur for p in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cote = max(max(xs) - min(xs), max(ys) - min(ys)) * marge
    return cx, cy, cote


def poser(image, meme, centre, hauteur, age):
    """Draws `meme` over `image` (in place), centred on `centre` (the face, smoothed), at a
    fixed `hauteur` in pixels: every meme has the same height, the width follows its format
    (never cropped: Absolute Cinema's hands are at its edges), and the size never changes
    with the face (steadier). Pop-in during the first POP seconds (`age`), thin
    white frame."""
    cx, cy = centre
    progres = min(1.0, age / POP)
    echelle = 0.8 + 0.2 * (1 - (1 - progres) ** 2)  # ease-out: fast start, soft landing
    mh, mw = meme.shape[:2]
    s = min(hauteur / mh, 0.95 * image.shape[1] / mw) * echelle  # never wider than the frame
    lw, lh = int(mw * s), int(mh * s)
    if min(lw, lh) < 20:
        return
    img = cv2.resize(meme, (lw, lh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (lw - 1, lh - 1), (255, 255, 255), max(3, lw // 70))
    h, w = image.shape[:2]
    x0, y0 = int(cx - lw / 2), int(cy - lh / 2)
    # Clip to the frame.
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + lw), min(h, y0 + lh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    image[dy0:dy1, dx0:dx1] = img[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
