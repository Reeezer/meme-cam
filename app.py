"""Mèmes en visio: a gesture in front of the webcam puts the matching meme over your face.

    uv run app.py                       preview window (screen-record it)
    uv run app.py --hud                 + what the camera detects: face box, hand dots (brand colours)
    uv run app.py --debug               + the raw numbers the rules read, to tune gestes.py
    uv run app.py --visio               + virtual webcam for Google Meet / Zoom (OBS installed)
    uv run app.py --visio --visio-miroir   the same, flipped: Meet mirrors your own preview, this puts the memes back the right way
    uv run app.py --enregistrer x.mp4   + records the result (real speed, H.264)
    uv run app.py --brut x.mp4          + records the plain webcam, to tune offline (analyser.py)
    uv run app.py --source essais.mp4 --hud --enregistrer demo.mp4   replays a recording
    uv run app.py --taille 0.7          meme height, as a share of the frame height (default 0.6)
Q or Échap to quit.
"""
import argparse
import sys
import time

import cv2

import hud
import memes
from detection import Detecteur
from enregistreur import Enregistreur
from gestes import Stabilisateur, mesures

GARDE_VISAGE = 1.5  # seconds a lost face box is kept (a hand over the face hides it)
# A gesture must be held ENTREE frames (~0.25 s at 30 ips) before its meme shows: shorter
# flashes appeared while the hands moved from one gesture to the next.
ENTREE, SORTIE = 8, 8
FENETRE = "meme-cam"
LISSAGE = 0.2  # share of the way the meme moves toward the face each frame: follows, without jitter


def texte_debug(image, lecture, geste_affiche):
    m = mesures(lecture.mains, lecture.visage)
    lignes = [f"brut: {lecture.geste or '-'}   affiche: {geste_affiche or '-'}"]
    lignes += [f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in m.items()]
    for k, l in enumerate(lignes):
        y = 90 + k * 30
        cv2.putText(image, l, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, l, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", metavar="FICHIER.mp4", help="replay a recording instead of the webcam (no window)")
    ap.add_argument("--largeur", type=int, default=1280)
    ap.add_argument("--hauteur", type=int, default=720)
    ap.add_argument("--taille", type=float, default=0.6, help="meme height, share of the frame height")
    ap.add_argument("--visio", action="store_true", help="virtual webcam (OBS Virtual Camera)")
    ap.add_argument("--visio-miroir", action="store_true",
                    help="flip what the virtual webcam sends: Meet shows your own preview mirrored, this cancels it")
    ap.add_argument("--enregistrer", metavar="FICHIER.mp4")
    ap.add_argument("--brut", metavar="FICHIER.mp4", help="plain webcam recording, for analyser.py")
    ap.add_argument("--hud", action="store_true", help="show what is detected (for the video)")
    ap.add_argument("--debug", action="store_true", help="show the numbers the rules read")
    ap.add_argument("--miroir", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    images, manquants = memes.charger()
    for m in manquants:
        print(f"Mème manquant : {m}")
    if not images:
        sys.exit("Aucun mème chargé : ajoute les images dans memes/ (voir memes/memes.json).")

    if args.source:
        # A --brut recording is already mirrored; timestamps come from the frame count.
        cam, args.miroir = cv2.VideoCapture(args.source), False
        fps_source = cam.get(cv2.CAP_PROP_FPS) or 30
    else:
        cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.largeur)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.hauteur)
    ok, image = cam.read()
    if not ok:
        sys.exit(f"Source introuvable : {args.source or f'webcam {args.camera}'}")
    h, w = image.shape[:2]
    if args.source:
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    virtuelle = None
    if args.visio:
        try:
            import pyvirtualcam
            virtuelle = pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Webcam virtuelle : {virtuelle.device} (choisis-la dans Meet / Zoom)")
        except Exception as e:  # no OBS Virtual Camera driver
            print(f"Webcam virtuelle indisponible ({e}). Installe OBS Studio, qui fournit le pilote.")
    reel = not args.source
    sortie = Enregistreur(args.enregistrer, w, h, temps_reel=reel) if args.enregistrer else None
    brut = Enregistreur(args.brut, w, h, temps_reel=reel) if args.brut else None

    det = Detecteur()
    stable = Stabilisateur(entree=ENTREE, sortie=SORTIE)
    centre, vue_a = None, 0.0
    geste_affiche, debut_geste = None, 0.0
    t0, n_images, fps, index = time.monotonic(), 0, 0.0, 0

    try:
        while True:
            ok, image = cam.read()
            if not ok:
                break
            if args.miroir:
                image = cv2.flip(image, 1)
            if brut:
                brut.ecrire(image)
            # Clock: real time live, frame count when replaying (processing is slower).
            now = t0 + index / fps_source if args.source else time.monotonic()
            index += 1
            lecture = det.lire(image, int((now - t0) * 1000))
            if lecture.points_visage:
                cx, cy, _ = memes.boite_visage(lecture.points_visage, w, h)
                centre = (cx, cy) if centre is None else (centre[0] + LISSAGE * (cx - centre[0]), centre[1] + LISSAGE * (cy - centre[1]))
                vue_a = now
            elif centre is not None and now - vue_a > GARDE_VISAGE:
                centre = None

            geste = stable(lecture.geste)
            if geste != geste_affiche:
                geste_affiche, debut_geste = geste, now
            meme_pose = geste_affiche in images and centre is not None
            if args.hud:  # detection first, the meme on top of it
                if lecture.points_visage and not meme_pose:
                    hud.cadre_visage(image, lecture.points_visage)
                hud.points_mains(image, lecture.mains)
            if meme_pose:
                memes.poser(image, images[geste_affiche], centre, args.taille * h, now - debut_geste)
            n_images += 1
            fps = n_images / max(1e-3, now - t0) if not args.source else fps_source
            if args.debug:
                texte_debug(image, lecture, geste_affiche)

            if virtuelle:
                virtuelle.send(cv2.flip(image, 1) if args.visio_miroir else image)
            if sortie:
                sortie.ecrire(image)
            if not args.source:
                cv2.imshow(FENETRE, image)
                if n_images % 15 == 0:  # live stats in the title bar
                    cv2.setWindowTitle(FENETRE, f"Meme-Cam · {fps:.1f} ips")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cam.release()
        for e in (sortie, brut):
            if e:
                e.fermer()
        if virtuelle:
            virtuelle.close()
        cv2.destroyAllWindows()
        for nom in (args.enregistrer, args.brut):
            if nom:
                print(f"Enregistré : {nom}")


if __name__ == "__main__":
    main()
