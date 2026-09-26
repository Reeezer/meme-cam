"""What the camera detects: a box around the face and small dots on the hands (no skeleton),
in a warm yellow / cream palette. No text on screen, so it films cleanly."""
import cv2

JAUNE = (63, 210, 255)  # #FFD23F, brand yellow (BGR)
CREME = (220, 233, 243)  # #F3E9DC, brand cream: fingertips
BOUTS = (4, 8, 12, 16, 20)


def points_mains(image, mains):
    h, w = image.shape[:2]
    for main in mains:
        for k, p in enumerate(main):
            cv2.circle(image, (int(p.x * w), int(p.y * h)), 4 if k in BOUTS else 3, CREME if k in BOUTS else JAUNE, -1, cv2.LINE_AA)


def boite(points):
    """Face box in normalised coordinates (x, y, w, h)."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def cadre_visage(image, points):
    h, w = image.shape[:2]
    x, y, bw, bh = boite(points)
    cv2.rectangle(image, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), JAUNE, 2)
