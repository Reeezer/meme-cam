"""Records frames through ffmpeg with their real timestamps: the webcam + detection runs at
15-25 frames/s, not 30, and a fixed-rate recorder would play back too fast."""
import subprocess
import time


class Enregistreur:
    def __init__(self, fichier, largeur, hauteur, fps=30, temps_reel=True):
        # Live: frames carry their arrival time. Replaying a file: a constant rate.
        entree = ["-use_wallclock_as_timestamps", "1"] if temps_reel else ["-r", str(fps)]
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{largeur}x{hauteur}",
                *entree, "-i", "-",
                "-fps_mode", "cfr", "-r", str(fps),
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", fichier,
            ],
            stdin=subprocess.PIPE,
        )

    def ecrire(self, image):
        self.proc.stdin.write(image.tobytes())

    def fermer(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
