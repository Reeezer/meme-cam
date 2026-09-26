"""Replays a plain webcam recording (app.py --brut) through the detector and prints, every
0.25 s, the gesture read and the numbers behind it: the data to tune gestes.py on real hands.

    uv run analyser.py brut.mp4
"""
import sys

import cv2

from detection import Detecteur
from gestes import mesures

PAS = 0.25


def main(fichier):
    vid = cv2.VideoCapture(fichier)
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    det = Detecteur()
    n, prochain = 0, 0.0
    while True:
        ok, image = vid.read()
        if not ok:
            break
        t = n / fps
        lecture = det.lire(image, int(t * 1000))
        n += 1
        if t < prochain:
            continue
        prochain += PAS
        m = mesures(lecture.mains, lecture.visage)
        chiffres = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
        print(f"{t:6.2f}s  {lecture.geste or '-':9}  mains={len(lecture.mains)}  {chiffres}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
