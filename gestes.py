"""Gesture recognition from MediaPipe landmarks: pure functions, no camera, testable.

Coordinates are MediaPipe's normalised image coordinates (x right, y down, 0-1). Every
distance is measured in face widths (temple to temple), so the rules hold whatever the
distance to the camera.
"""
from dataclasses import dataclass
from math import dist

# MediaPipe hand landmark indices
POIGNET = 0
BOUTS = {"pouce": 4, "index": 8, "majeur": 12, "annulaire": 16, "auriculaire": 20}
PHALANGES = {"pouce": 3, "index": 6, "majeur": 10, "annulaire": 14, "auriculaire": 18}
PAUME = (0, 5, 9, 13, 17)

# MediaPipe face mesh indices
NEZ, FRONT, MENTON = 1, 10, 152
TEMPE_G, TEMPE_D = 234, 454
OEIL_G, OEIL_D = 159, 386

GESTES = ("coeur", "priere", "cinema", "drake", "pointe", "rollsafe", "main_visage")


@dataclass
class Visage:
    nez: tuple
    front: tuple
    menton: tuple
    tempe_g: tuple
    tempe_d: tuple
    yeux: tuple  # midpoint between the eyes
    yeux_fermes: float | None  # 0-1 (eyeBlink blendshapes), None when unknown (body model)

    @property
    def largeur(self):
        return dist(self.tempe_g, self.tempe_d)

    @classmethod
    def depuis_mediapipe(cls, points, blendshapes=None):
        p = lambda i: (points[i].x, points[i].y)
        fermes = 0.0
        if blendshapes:
            scores = {b.category_name: b.score for b in blendshapes}
            fermes = (scores.get("eyeBlinkLeft", 0) + scores.get("eyeBlinkRight", 0)) / 2
        g, d = p(OEIL_G), p(OEIL_D)
        return cls(p(NEZ), p(FRONT), p(MENTON), p(TEMPE_G), p(TEMPE_D), ((g[0] + d[0]) / 2, (g[1] + d[1]) / 2), fermes)

    @classmethod
    def depuis_corps(cls, points):
        """Face estimated from the body model's head points, for when the face model loses a
        face that is too far from the camera. Temples: halfway between the eye's outer corner
        and the ear; forehead and chin: extrapolated from the eyes-mouth axis. Eye state is
        unknown."""
        p = lambda i: (points[i].x, points[i].y)
        mil = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        yeux = mil(p(2), p(5))
        bouche = mil(p(9), p(10))
        axe = (yeux[0] - bouche[0], yeux[1] - bouche[1])
        front = (yeux[0] + 0.9 * axe[0], yeux[1] + 0.9 * axe[1])
        menton = (bouche[0] - 0.8 * axe[0], bouche[1] - 0.8 * axe[1])
        return cls(p(0), front, menton, mil(p(3), p(7)), mil(p(6), p(8)), yeux, None)


def _xy(pt):
    return (pt[0], pt[1]) if isinstance(pt, tuple) else (pt.x, pt.y)


def _main(points):
    """Normalises a MediaPipe hand (21 landmarks) into (x, y) tuples."""
    return [_xy(p) for p in points]


def doigts_tendus(main):
    """Names of the extended fingers: a finger is extended when its tip is clearly farther
    from the wrist than its middle joint (works in any hand orientation)."""
    w = main[POIGNET]
    tendus = set()
    for nom, bout in BOUTS.items():
        seuil = 1.15 if nom != "pouce" else 1.05
        if dist(main[bout], w) > seuil * dist(main[PHALANGES[nom]], w):
            tendus.add(nom)
    return tendus


def centre_paume(main):
    xs = [main[i][0] for i in PAUME]
    ys = [main[i][1] for i in PAUME]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _index_seul(tendus):
    return "index" in tendus and not ({"majeur", "annulaire", "auriculaire"} & tendus)


def _ouverte(tendus):
    return len(tendus - {"pouce"}) >= 3


def index_dominant(main):
    """The index reaches clearly farther from the wrist than the other long fingers. Holds
    when pointing even if the folded fingers look half-open from the camera (a plain
    'index only' test failed on real pointing)."""
    w = main[POIGNET]
    i = dist(main[BOUTS["index"]], w)
    autres = max(dist(main[BOUTS[n]], w) for n in ("majeur", "annulaire", "auriculaire"))
    return i > 1.25 * autres


def forme_pointee(main):
    """Pointing hand seen from the side (arm stretched sideways): the middle finger can look
    as long as the index, but ring and little fingers are much shorter (measured 0.5-0.65 of
    the index), whereas on an open hand they are nearly as long."""
    w = main[POIGNET]
    l = {n: dist(main[BOUTS[n]], w) for n in ("index", "majeur", "annulaire", "auriculaire")}
    return l["index"] >= 0.95 * l["majeur"] and l["annulaire"] < 0.75 * l["index"] and l["auriculaire"] < 0.65 * l["index"]


def doigts_vers_le_haut(main, S):
    """Fingers point up: the middle fingertip is well above the wrist."""
    return main[POIGNET][1] - main[BOUTS["majeur"]][1] > 0.5 * S


# Thresholds calibrated on a real recording (~2.5 m from the webcam,
# face found by the body model): each sits between what the gesture measured and what the
# other gestures measured. Units: face widths.
def reconnaitre(mains, visage):
    """Returns one of GESTES or None. `mains`: list of 0-2 hands (21 landmarks each);
    `visage`: a Visage, or None when no face is visible (then nothing is recognised: every
    rule is relative to the face)."""
    if visage is None or not mains:
        return None
    S = visage.largeur
    if S <= 0:
        return None
    mains = [_main(m) for m in mains]
    tendus = [doigts_tendus(m) for m in mains]
    paumes = [centre_paume(m) for m in mains]
    centre_x = (visage.tempe_g[0] + visage.tempe_d[0]) / 2
    sous_menton = lambda p: (p[1] - visage.yeux[1]) / S > 1.5

    # ── Two hands ──
    if len(mains) == 2:
        # Heart (Bieber): index tips touch at the top, thumb tips at the bottom, and the wrists
        # stay apart (in prayer they touch: measured 0.82-0.89). Not yet calibrated on a real
        # recording: thresholds to check.
        index_joints = dist(mains[0][BOUTS["index"]], mains[1][BOUTS["index"]]) < 0.5 * S
        pouces_joints = dist(mains[0][BOUTS["pouce"]], mains[1][BOUTS["pouce"]]) < 0.6 * S
        poignets_ecartes = dist(mains[0][POIGNET], mains[1][POIGNET]) > 1.1 * S
        index_en_haut = all(m[BOUTS["index"]][1] < m[BOUTS["pouce"]][1] for m in mains)
        if index_joints and pouces_joints and poignets_ecartes and index_en_haut:
            return "coeur"
        bouts_proches = dist(mains[0][BOUTS["index"]], mains[1][BOUTS["index"]]) < 0.5 * S  # measured 0.10-0.17
        poignets_proches = dist(mains[0][POIGNET], mains[1][POIGNET]) < 1.1 * S  # measured 0.82-0.89
        if bouts_proches and poignets_proches and all(sous_menton(p) for p in paumes):
            return "priere"
        de_part_et_dautre = min(p[0] for p in paumes) < centre_x - 0.6 * S and max(p[0] for p in paumes) > centre_x + 0.6 * S
        a_hauteur = all(abs(p[1] - visage.nez[1]) < 1.2 * S for p in paumes)
        if all(_ouverte(t) for t in tendus) and de_part_et_dautre and a_hauteur:
            return "cinema"

    # ── One hand (the one closest to the face decides) ──
    i = min(range(len(mains)), key=lambda k: dist(paumes[k], visage.nez))
    main, t, paume = mains[i], tendus[i], paumes[i]
    bout_index = main[BOUTS["index"]]
    ecart = abs(paume[0] - centre_x) / S
    hauteur = abs(paume[1] - visage.nez[1]) / S
    index_nez = dist(bout_index, visage.nez) / S
    index_tete = min(dist(bout_index, visage.tempe_g), dist(bout_index, visage.tempe_d), dist(bout_index, visage.front)) / S

    # Index clearly the longest finger: Roll Safe near the head, DiCaprio away from it.
    if index_dominant(main) or _index_seul(t) or forme_pointee(main):
        if index_nez < 1.7 and index_tete < 0.9:  # measured nez 1.0-1.5, tête 0.45-0.78
            return "rollsafe"
        # Pointing is done with a raised arm (palm measured 1.2-1.9 below the eyes); a hand
        # resting in the lap (4-5.5 below) must not count.
        if index_nez > 1.9 and (paume[1] - visage.yeux[1]) / S < 3.0:  # index-nose measured 2.6-3.4
            return "pointe"
        return None
    # Open hand over the upper face (fingers on the forehead, palm over the eyes or lower):
    # Picard and Jonah Hill are the same gesture for a camera, so one meme.
    if len(t - {"pouce"}) == 4 and ecart < 0.6 and index_tete < 1.2 and dist(paume, visage.yeux) / S < 2.0:
        return "main_visage"  # measured ecart 0.15-0.26, tête 0.86-1.12, paume-yeux 1.6-1.9
    # Hands pressed together, seen as one open hand pointing up, centred below the chin.
    if _ouverte(t) and doigts_vers_le_haut(main, S) and ecart < 0.7 and sous_menton(paume):
        return "priere"  # measured ecart 0.4, 4.3 below the eyes
    # One open palm raised beside the face, fingers up: Drake "no".
    if _ouverte(t) and doigts_vers_le_haut(main, S) and 0.7 < ecart < 1.9 and hauteur < 0.6:
        return "drake"  # measured ecart 1.04-1.08, hauteur 0.09-0.16
    return None


def mesures(mains, visage):
    """The numbers the rules look at, in face widths, for the hand closest to the face: what
    `--debug` shows and `analyser.py` prints, to tune thresholds on real hands."""
    if visage is None or not mains or visage.largeur <= 0:
        return {}
    S = visage.largeur
    mains = [_main(m) for m in mains]
    paumes = [centre_paume(m) for m in mains]
    i = min(range(len(mains)), key=lambda k: dist(paumes[k], visage.nez))
    main, paume = mains[i], paumes[i]
    bout = main[BOUTS["index"]]
    r = {
        "doigts": "".join(n[0] for n in BOUTS if n in doigts_tendus(main)),  # p i m a o
        "paume_yeux": dist(paume, visage.yeux) / S,
        "index_tempe": min(dist(bout, visage.tempe_g), dist(bout, visage.tempe_d)) / S,
        "index_nez": dist(bout, visage.nez) / S,
        "index_front": dist(bout, visage.front) / S,
        "paume_sous_yeux": (paume[1] - visage.yeux[1]) / S,
        "ecart_x": abs(paume[0] - (visage.tempe_g[0] + visage.tempe_d[0]) / 2) / S,
        "hauteur": abs(paume[1] - visage.nez[1]) / S,
        "yeux_fermes": -1.0 if visage.yeux_fermes is None else visage.yeux_fermes,
    }
    if len(mains) == 2:
        r["bouts_index"] = dist(mains[0][BOUTS["index"]], mains[1][BOUTS["index"]]) / S
        r["poignets"] = dist(mains[0][POIGNET], mains[1][POIGNET]) / S
        r["doigts2"] = "".join(n[0] for n in BOUTS if n in doigts_tendus(mains[1 - i]))
    return r


class Stabilisateur:
    """A gesture shows up only after it has been seen `entree` frames in a row, and stays
    `sortie` frames after it stops: no flicker when the detector hesitates for one frame."""

    def __init__(self, entree=4, sortie=8):
        self.entree, self.sortie = entree, sortie
        self.actif, self.candidat, self.vu, self.absent = None, None, 0, 0

    def __call__(self, geste):
        if geste == self.candidat:
            self.vu += 1
        else:
            self.candidat, self.vu = geste, 1
        if geste is not None and self.vu >= self.entree:
            self.actif, self.absent = geste, 0
        elif geste != self.actif:
            self.absent += 1
            if self.absent >= self.sortie:
                self.actif = None
        else:
            self.absent = 0
        return self.actif
