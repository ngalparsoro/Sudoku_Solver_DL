"""Recorta el tablero de la foto y lo parte en sus 81 casillas (9x9).

Flujo:
    foto
      -> SudokuDetector localiza la caja del tablero (YOLO)
      -> se afinan las 4 esquinas reales dentro de esa caja (contorno)
      -> warpPerspective: vista cuadrada de frente (size x size)
      -> split en 9x9 = 81 casillas
      -> (opcional) guardar cada casilla como imagen

Uso por linea de comandos:
    python src/cropper.py foto.jpg salida/    # guarda board.png + 81 celdas
"""

from pathlib import Path

import numpy as np
import cv2

from detector import SudokuDetector

ROOT = Path(__file__).resolve().parent.parent


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray, size: int = 450) -> tuple[np.ndarray, np.ndarray]:
    """Rectifica `image` a una vista cuadrada `size`x`size` dadas 4 esquinas.

    Devuelve (imagen_rectificada, M). M sirve para volver a la foto (perspectiva inversa).
    """
    rect = order_points(pts)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (size, size)), M


def crop_bbox(image: np.ndarray, xyxy, pad: int = 10) -> tuple[np.ndarray, tuple[int, int]]:
    """Recorta la caja (x1,y1,x2,y2) con margen. Devuelve (recorte, offset (ox,oy))."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy[:4])
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return image[y1:y2, x1:x2], (x1, y1)


def find_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Las 4 esquinas del tablero como el mayor contorno cuadrilatero, o None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(biggest, 0.02 * cv2.arcLength(biggest, True), True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype("float32")


def _tablero_plausible(corners: np.ndarray, bbox, shape) -> bool:
    """True si el cuadrilatero de la imagen completa parece el tablero real:
    cuadrado, grande y consistente con donde YOLO vio algo. Sirve para corregir
    cajas de YOLO que se quedan cortas (p.ej. sudokus digitales/sombreados) sin
    arriesgarse a coger un rectangulo cualquiera en una foto.
    """
    (x1, y1), (x2, y2) = corners.min(0), corners.max(0)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    if not (0.8 <= w / h <= 1.25):                       # ha de ser ~cuadrado
        return False
    area_full = w * h
    H, W = shape[:2]
    if area_full < 0.08 * W * H:                         # y razonablemente grande
        return False
    bx1, by1, bx2, by2 = bbox[:4]
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):          # ha de contener la caja YOLO
        return False
    area_yolo = max(1.0, (bx2 - bx1) * (by2 - by1))
    # ni igual ni desproporcionado. El tope alto es generoso porque en capturas
    # digitales YOLO a veces solo caza una FRANJA del tablero (p.ej. 3 filas), y
    # entonces el cuadrado real es ~3x esa caja; con 2.5 se rechazaba por error.
    return 0.7 <= area_full / area_yolo <= 4.0


def warp_board(image: np.ndarray, bbox=None, size: int = 450) -> tuple[np.ndarray, np.ndarray]:
    """Rectifica el tablero. Si se pasa bbox (de YOLO) se afinan las esquinas dentro;
    si no, se busca en la imagen completa. Devuelve (tablero, M).
    """
    # 1) si YOLO dio caja pero el cuadrilatero de la imagen COMPLETA es un tablero
    #    plausible (cuadrado y coherente con YOLO), lo preferimos: corrige cajas de
    #    YOLO que se quedan cortas en capturas digitales.
    if bbox is not None:
        full = find_board_corners(image)
        if full is not None and _tablero_plausible(full, bbox, image.shape):
            return four_point_transform(image, full, size)

    region, (ox, oy) = (crop_bbox(image, bbox) if bbox is not None else (image, (0, 0)))
    corners = find_board_corners(region)
    if corners is not None:
        corners += np.array([ox, oy], dtype="float32")
    elif bbox is not None:
        # sin cuadrilatero claro: caemos a las esquinas de la propia caja YOLO
        x1, y1, x2, y2 = bbox[:4]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
    else:
        raise ValueError("No se encontro el tablero")
    return four_point_transform(image, corners, size)


def split_cells(board: np.ndarray, grid: int = 9) -> list[list[np.ndarray]]:
    """Divide el tablero rectificado en grid x grid casillas (lista de listas)."""
    h, w = board.shape[:2]
    ch, cw = h // grid, w // grid
    return [[board[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] for c in range(grid)] for r in range(grid)]


def save_cells(cells: list[list[np.ndarray]], out_dir: Path | str) -> int:
    """Guarda cada casilla como cell_r{r}_c{c}.png en out_dir. Devuelve nº guardadas."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            cv2.imwrite(str(out / f"cell_r{r}_c{c}.png"), cell)
            n += 1
    return n


class SudokuCropper:
    """Recorta el tablero y lo parte en 81 casillas."""

    def __init__(self, det: SudokuDetector | None = None, size: int = 450):
        self.detector = det or SudokuDetector()
        self.size = size

    def crop_board(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (tablero_rectificado, M). Lanza error si no se detecta tablero."""
        det = self.detector.detect(image)
        bbox = det[:4] if det is not None else None
        return warp_board(image, bbox, self.size)

    def to_cells(self, image: np.ndarray) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        """Devuelve (tablero_rectificado, casillas 9x9)."""
        board, _ = self.crop_board(image)
        return board, split_cells(board)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python src/cropper.py <foto.jpg> <carpeta_salida>")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("No se pudo leer la imagen:", sys.argv[1])
        raise SystemExit(1)
    cropper = SudokuCropper()
    board, cells = cropper.to_cells(img)
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "board.png"), board)
    n = save_cells(cells, out_dir / "cells")
    print(f"Tablero guardado en {out_dir/'board.png'} y {n} casillas en {out_dir/'cells'}")
