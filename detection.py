"""One frame in, what MediaPipe sees and which gesture the rules read. Shared by the live
app and by analyser.py, so tuning on a recorded clip matches what happens live."""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from gestes import Visage, reconnaitre

MODELES = Path(__file__).parent / "modeles"
BASE = "https://storage.googleapis.com/mediapipe-models"
URLS = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def modele(nom: str) -> str:
    """Path to a MediaPipe model, downloaded on first use (about 15 MB in total)."""
    chemin = MODELES / nom
    if not chemin.exists():
        MODELES.mkdir(exist_ok=True)
        print(f"Téléchargement du modèle {nom}…")
        urllib.request.urlretrieve(URLS[nom], chemin)
    return str(chemin)


@dataclass
class Lecture:
    mains: list  # MediaPipe hand landmarks (0-2 hands)
    points_visage: list | None  # face landmarks (478 from the face model, or 11 head points)
    visage: Visage | None
    geste: str | None  # raw, before stabilisation
    source: str = "aucun"  # "visage" (face model), "corps" (body model, far away) or "aucun"


class Detecteur:
    def __init__(self):
        video = vision.RunningMode.VIDEO
        self.mains = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modele("hand_landmarker.task")),
            running_mode=video, num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.visage = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modele("face_landmarker.task")),
            running_mode=video, num_faces=1, output_face_blendshapes=True))
        # The face model only sees faces close to the camera (video-call distance). Beyond
        # ~1.5 m the body model still finds the head: it takes over when the face is lost.
        self.corps = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modele("pose_landmarker_lite.task")),
            running_mode=video, num_poses=1))

    def lire(self, image_bgr, ts_ms: int) -> Lecture:
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        rm = self.mains.detect_for_video(rgb, ts_ms)
        rv = self.visage.detect_for_video(rgb, ts_ms)
        rc = self.corps.detect_for_video(rgb, ts_ms)  # every frame: VIDEO mode needs a continuous stream
        points, visage, source = None, None, "aucun"
        if rv.face_landmarks:
            points = rv.face_landmarks[0]
            visage = Visage.depuis_mediapipe(points, rv.face_blendshapes[0] if rv.face_blendshapes else None)
            source = "visage"
        elif rc.pose_landmarks:
            points = rc.pose_landmarks[0][:11]  # head points: nose, eyes, ears, mouth
            visage = Visage.depuis_corps(points)
            source = "corps"
        return Lecture(rm.hand_landmarks, points, visage, reconnaitre(rm.hand_landmarks, visage), source)
