"""Deteccion del marco de un sudoku en una foto, con YOLO (Ultralytics).

Expone:
  - SudokuDetector: carga un modelo entrenado y detecta la caja del tablero.
  - train_detector: entrena YOLO sobre el dataset y guarda el mejor modelo.

Uso por linea de comandos (requiere modelo entrenado en models/sudoku_detector.pt):
    python src/detector.py foto.jpg        # guarda detected.png
"""

from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "sudoku.yaml"
DEFAULT_WEIGHTS = ROOT / "models" / "sudoku_detector.pt"
BASE_MODEL = ROOT / "models" / "yolov8n.pt"            # pesos base de Ultralytics (punto de partida)

# Caja del tablero como (x1, y1, x2, y2, confianza) en pixeles de la imagen.
Detection = tuple[float, float, float, float, float]


class SudokuDetector:
    """Detecta el marco del sudoku usando un modelo YOLO entrenado."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, device: str = "mps"):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No existe el modelo {weights}. Entrenalo antes con train_detector "
                "(ver notebooks/detector_dev.ipynb)."
            )
        self.model = YOLO(str(weights))
        self.device = device

    def detect(self, image: np.ndarray, conf: float = 0.25) -> Detection | None:
        """Devuelve la caja del tablero mas probable, o None si no detecta nada."""
        res = self.model.predict(image, conf=conf, device=self.device, verbose=False)[0]
        if len(res.boxes) == 0:
            return None
        best = res.boxes[int(res.boxes.conf.argmax())]
        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        return (x1, y1, x2, y2, float(best.conf[0]))

    def detect_and_draw(self, image: np.ndarray, conf: float = 0.25) -> tuple[np.ndarray, Detection | None]:
        """Como detect, pero devuelve tambien la imagen con la caja dibujada."""
        out = image.copy()
        det = self.detect(image, conf)
        if det is None:
            return out, None
        x1, y1, x2, y2, c = det
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(out, f"sudoku {c:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return out, det


def train_detector(data_yaml: Path | str = DATA_YAML, base_model: str | Path = BASE_MODEL,
                   epochs: int = 60, imgsz: int = 640, batch: int = 16,
                   device: str = "mps", name: str = "sudoku_detector",
                   patience: int = 15):
    """Entrena YOLO para detectar el tablero y copia el mejor modelo a models/.

    Devuelve (ruta_modelo, results).
    """
    from ultralytics import YOLO

    model = YOLO(str(base_model))
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                          batch=batch, device=device, name=name, patience=patience)
    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = ROOT / "models" / "sudoku_detector.pt"
    dst.parent.mkdir(exist_ok=True)
    dst.write_bytes(best.read_bytes())
    print(f"Mejor modelo copiado a: {dst}")
    return dst, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/detector.py <foto.jpg>")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("No se pudo leer la imagen:", sys.argv[1])
        raise SystemExit(1)
    detector = SudokuDetector()
    out, det = detector.detect_and_draw(img)
    print("Deteccion (x1,y1,x2,y2,conf):", det)
    cv2.imwrite("detected.png", out)
    print("Guardado detected.png")
