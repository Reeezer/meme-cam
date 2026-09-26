"""Each gesture is built as a synthetic hand around a synthetic face, then recognised."""
import math

import pytest

from gestes import Stabilisateur, Visage, doigts_tendus, reconnaitre

# A face centred at x=0.5, 0.2 wide (temple to temple). y grows downwards.
VISAGE = Visage(
    nez=(0.5, 0.45), front=(0.5, 0.30), menton=(0.5, 0.60),
    tempe_g=(0.4, 0.40), tempe_d=(0.6, 0.40), yeux=(0.5, 0.38), yeux_fermes=0.0,
)
S = 0.2
DOIGTS = ("pouce", "index", "majeur", "annulaire", "auriculaire")


def main(poignet, angle_deg, tendus, taille=0.16):
    """21 landmarks, real proportions (wrist to fingertip ≈ 0.9 face width): the fingers fan out from the wrist in direction `angle_deg`
    (0 = up, 90 = right). An extended finger reaches far; a folded one stays close."""
    wx, wy = poignet
    pts = [(wx, wy)] * 21
    for k, nom in enumerate(DOIGTS):
        a = math.radians(angle_deg + (k - 2) * 12)
        ux, uy = math.sin(a), -math.cos(a)
        base = 0.45 * taille
        # MCP, PIP, DIP, TIP indices for each finger
        idx = {"pouce": (1, 2, 3, 4), "index": (5, 6, 7, 8), "majeur": (9, 10, 11, 12),
               "annulaire": (13, 14, 15, 16), "auriculaire": (17, 18, 19, 20)}[nom]
        longueurs = (base, base + 0.25 * taille, base + 0.45 * taille, base + 0.65 * taille) if nom in tendus \
            else (base, base + 0.2 * taille, base + 0.15 * taille, base + 0.1 * taille)
        for j, l in zip(idx, longueurs):
            pts[j] = (wx + ux * l, wy + uy * l)
    return pts


def main_bout_index_en(point, angle_deg, tendus, taille=0.16):
    """Same hand, translated so that the index fingertip lands on `point`."""
    m = main((0, 0), angle_deg, tendus, taille)
    dx, dy = point[0] - m[8][0], point[1] - m[8][1]
    return [(x + dx, y + dy) for x, y in m]


def main_paume_en(point, angle_deg, tendus, taille=0.16):
    m = main((0, 0), angle_deg, tendus, taille)
    cx = sum(m[i][0] for i in (0, 5, 9, 13, 17)) / 5
    cy = sum(m[i][1] for i in (0, 5, 9, 13, 17)) / 5
    return [(x + point[0] - cx, y + point[1] - cy) for x, y in m]


OUVERTE = set(DOIGTS)
INDEX = {"index"}


def test_doigts_tendus_lit_une_main_ouverte_et_un_index():
    assert doigts_tendus(main((0.5, 0.8), 0, OUVERTE)) == OUVERTE
    assert doigts_tendus(main((0.5, 0.8), 0, INDEX)) == INDEX


def test_priere_deux_mains_jointes_devant_la_poitrine():
    """As performed on a real recording: hands together at chest height."""
    g = main_bout_index_en((0.49, 0.75), 20, OUVERTE)
    d = main_bout_index_en((0.51, 0.75), -20, OUVERTE)
    assert reconnaitre([g, d], VISAGE) == "priere"


def _coeur():
    """Two hands at chest height, wrists apart, index tips meeting at the top and thumb tips
    meeting at the bottom."""
    g = main((0.36, 0.86), 45, OUVERTE)
    d = main((0.64, 0.86), -45, OUVERTE)
    g[8], d[8] = (0.495, 0.70), (0.505, 0.70)  # index tips
    g[4], d[4] = (0.49, 0.82), (0.51, 0.82)  # thumb tips
    return [g, d]


def test_coeur_index_et_pouces_joints_poignets_ecartes():
    assert reconnaitre(_coeur(), VISAGE) == "coeur"


def test_priere_n_est_pas_un_coeur():
    """Prayer: wrists close together, so never read as a heart."""
    g = main_bout_index_en((0.49, 0.75), 20, OUVERTE)
    d = main_bout_index_en((0.51, 0.75), -20, OUVERTE)
    assert reconnaitre([g, d], VISAGE) == "priere"


def test_cinema_deux_mains_ouvertes_de_part_et_dautre():
    g = main_paume_en((0.30, 0.45), -20, OUVERTE)
    d = main_paume_en((0.70, 0.45), 20, OUVERTE)
    assert reconnaitre([g, d], VISAGE) == "cinema"


def test_drake_une_paume_ouverte_a_cote_du_visage():
    assert reconnaitre([main_paume_en((0.68, 0.47), 0, OUVERTE)], VISAGE) == "drake"


def test_pointe_bras_tendu_sur_le_cote():
    """As performed on the real recording: arm stretched sideways, index far from the head."""
    assert reconnaitre([main_bout_index_en((0.08, 0.55), -90, INDEX)], VISAGE) == "pointe"


def test_rollsafe_index_sur_la_tempe():
    assert reconnaitre([main_bout_index_en((0.61, 0.40), -30, INDEX)], VISAGE) == "rollsafe"


def test_main_sur_le_visage_paume_sur_les_yeux():
    """Picard and Jonah Hill are the same gesture for a camera: one meme."""
    assert reconnaitre([main_paume_en((0.5, 0.39), 0, OUVERTE)], VISAGE) == "main_visage"


def test_main_sur_le_visage_doigts_au_front_paume_basse():
    assert reconnaitre([main_bout_index_en((0.52, 0.31), -10, OUVERTE)], VISAGE) == "main_visage"


def test_priere_vue_comme_une_seule_main():
    """Hands pressed together are often detected as one hand: open, fingers up, centred,
    below the chin."""
    assert reconnaitre([main_paume_en((0.5, 0.75), 0, OUVERTE)], VISAGE) == "priere"


def test_pointer_depuis_les_genoux_ne_compte_pas():
    """A pointing-shaped hand resting in the lap is not DiCaprio."""
    assert reconnaitre([main_bout_index_en((0.5, 1.2), 180, INDEX)], VISAGE) is None


def test_rien_sans_visage_ni_main():
    assert reconnaitre([], VISAGE) is None
    assert reconnaitre([main((0.5, 0.8), 0, OUVERTE)], None) is None


def test_main_au_repos_sous_le_menton_ne_declenche_rien():
    assert reconnaitre([main((0.45, 0.95), 0, set())], VISAGE) is None


def test_stabilisateur_ignore_un_geste_d_une_image():
    s = Stabilisateur(entree=3, sortie=2)
    assert [s(g) for g in ["drake", None, None]] == [None, None, None]


@pytest.mark.parametrize("sequence, attendu", [
    (["drake"] * 3, "drake"),
    (["drake"] * 3 + [None], "drake"),
    (["drake"] * 3 + [None, None], None),
])
def test_stabilisateur_entree_et_sortie(sequence, attendu):
    s = Stabilisateur(entree=3, sortie=2)
    for g in sequence:
        r = s(g)
    assert r == attendu
