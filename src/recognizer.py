"""Reconocimiento de los digitos de las 81 casillas, con una CNN entrenada en MNIST.

Cada casilla se clasifica como:
    0      -> casilla vacia (detectada por cantidad de tinta)
    1..9   -> digito (CNN entrenada con MNIST)

Expone:
  - DigitCNN: la red (10 salidas, clases 0-9 de MNIST).
  - preprocess_cell: normaliza una casilla al formato MNIST (28x28, blanco sobre negro).
  - DigitRecognizer: carga el modelo y devuelve la lista de digitos de un tablero.
"""

from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "digit_cnn.pt"
CELL_SIZE = 28


class DigitCNN(nn.Module):
    """CNN compacta. 10 salidas (clases 0-9, como MNIST)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def preprocess_cell(cell: np.ndarray, out: int = CELL_SIZE, ink_thresh: float = 0.035):
    """Normaliza una casilla al formato MNIST: digito blanco centrado en `out`x`out`.

    Devuelve la imagen float32 [0,1], o None si la casilla esta vacia.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    h, w = gray.shape
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 7)
    thr = cv2.medianBlur(thr, 3)           # denoise (fondos lisos con ruido)
    # Quita las lineas de rejilla por morfologia, en ambas direcciones: detecta
    # los trazos que cruzan casi toda la casilla (las lineas) y los resta, dejando
    # solo el digito central. Hay que quitar vertical Y horizontal (solo una empeora).
    kf = 0.8
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(kf * h))))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kf * w)), 1))
    vlines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vk)
    hlines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, hk)
    # Las lineas de rejilla viven en los BORDES; un digito (sobre todo el "1", que
    # es un trazo casi vertical) vive en el CENTRO. Si restamos las lineas en toda
    # la casilla nos comemos los "1". Por eso solo restamos en las bandas de los
    # bordes: lineas verticales a izq/dcha, horizontales arriba/abajo.
    bw, bh = max(1, int(0.22 * w)), max(1, int(0.22 * h))
    vmask = np.zeros_like(thr); vmask[:, :bw] = 255; vmask[:, w - bw:] = 255
    hmask = np.zeros_like(thr); hmask[:bh, :] = 255; hmask[h - bh:, :] = 255
    lines = cv2.bitwise_or(cv2.bitwise_and(vlines, vmask),
                           cv2.bitwise_and(hlines, hmask))
    thr = cv2.subtract(thr, lines)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # quita motas
    m = max(1, int(0.10 * min(h, w)))      # recorte de margen tras limpiar lineas
    roi = thr[m:h - m, m:w - m]
    if roi.size == 0 or roi.mean() / 255 < ink_thresh:   # poca tinta -> vacia
        return None
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = roi.shape
    cands = [cv2.boundingRect(c) for c in cnts]
    cands = [b for b in cands if b[2] * b[3] >= 0.02 * roi.size]
    if not cands:
        return None
    max_area = max(b[2] * b[3] for b in cands)
    big = [b for b in cands if b[2] * b[3] >= 0.25 * max_area]
    x, y, bw, bh = min(big, key=lambda b: (b[0] + b[2] / 2 - W / 2) ** 2 + (b[1] + b[3] / 2 - H / 2) ** 2)
    # Rechaza restos de linea de rejilla: una astilla muy fina que ademas recorre
    # casi toda la casilla es una linea sobreviviente, no un digito (ni un "1", que
    # es mas ancho y no llega de borde a borde). Evita leer casillas VACIAS como 1/2.
    if (bw <= 0.18 * W and bh >= 0.70 * H) or (bh <= 0.18 * H and bw >= 0.70 * W):
        return None
    digit = roi[y:y + bh, x:x + bw]
    size = max(bw, bh)
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - bh) // 2, (size - bw) // 2
    canvas[oy:oy + bh, ox:ox + bw] = digit
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


class DigitRecognizer:
    """Lee los digitos de las casillas usando la CNN entrenada en MNIST."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, device: str = "cpu"):
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No existe el modelo {weights}. Entrenalo antes "
                "(ver notebooks/recognizer_dev.ipynb)."
            )
        self.device = device
        self.model = DigitCNN().to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict_cell(self, cell: np.ndarray) -> int:
        """0 si la casilla esta vacia, o el digito 1-9."""
        img = preprocess_cell(cell)
        if img is None:
            return 0
        x = torch.from_numpy(img)[None, None].to(self.device)
        logits = self.model(x)[0]
        return int(logits[1:].argmax().item()) + 1   # restringe a digitos 1-9

    def recognize(self, cells: list[list[np.ndarray]]) -> list[int]:
        """Procesa las 81 casillas (9x9) y devuelve una lista plana de 81 enteros
        en orden de lectura (fila a fila)."""
        return [self.predict_cell(cells[r][c]) for r in range(9) for c in range(9)]

    def recognize_grid(self, cells: list[list[np.ndarray]]) -> list[list[int]]:
        """Como recognize, pero devuelve la matriz 9x9."""
        flat = self.recognize(cells)
        return [flat[i * 9:(i + 1) * 9] for i in range(9)]

    def recognize_from_dir(self, folder: Path | str) -> list[int]:
        """Lee cell_r{r}_c{c}.png de una carpeta (las que genera cropper.save_cells)
        y devuelve la lista plana de 81 enteros."""
        folder = Path(folder)
        out = []
        for r in range(9):
            for c in range(9):
                cell = cv2.imread(str(folder / f"cell_r{r}_c{c}.png"))
                out.append(0 if cell is None else self.predict_cell(cell))
        return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/recognizer.py <carpeta_con_celdas>")
        raise SystemExit(1)
    rec = DigitRecognizer()
    lista = rec.recognize_from_dir(sys.argv[1])
    print("Lista de 81 digitos (0 = vacia):")
    print(lista)
