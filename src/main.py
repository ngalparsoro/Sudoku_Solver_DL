"""Pipeline completo: foto de un sudoku -> solucion.

Une los 4 pasos: detector (YOLO) -> cropper (81 casillas) -> recognizer (CNN)
-> solver (backtracking). Devuelve TODAS las etapas intermedias para poder
visualizarlas (pensado para la app de Streamlit).

Uso por linea de comandos:
    python src/main.py foto.jpg        # imprime la solucion y guarda solucion.png
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2

import detector
import cropper
import recognizer
import solver

ROOT = Path(__file__).resolve().parent.parent
Board = list[list[int]]


@dataclass
class Resultado:
    """Todas las etapas del pipeline, para inspeccion/visualizacion."""
    imagen: np.ndarray                      # foto original (BGR)
    deteccion: tuple | None = None          # (x1,y1,x2,y2,conf) o None
    imagen_anotada: np.ndarray | None = None  # foto con la caja del tablero
    tablero: np.ndarray | None = None       # tablero rectificado 450x450
    M: np.ndarray | None = None             # matriz de perspectiva (para overlay)
    leido: Board | None = None              # matriz 9x9 que leyo la red
    tablero_leido: np.ndarray | None = None  # tablero con los digitos leidos dibujados
    valido: bool = False                    # la lectura es un sudoku sin conflictos
    solucion: Board | None = None           # solucion del solver (o None)
    imagen_solucion: np.ndarray | None = None  # foto con la solucion superpuesta
    error: str | None = None                # mensaje si algo fallo


def dibujar_lectura(tablero: np.ndarray, grid: Board, size: int = 450) -> np.ndarray:
    """Dibuja sobre el tablero rectificado los digitos que leyo la red (en azul)."""
    out = tablero.copy()
    step = size // 9
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                org = (c * step + step // 3, r * step + 2 * step // 3)
                cv2.putText(out, str(grid[r][c]), org, cv2.FONT_HERSHEY_SIMPLEX,
                            step / 55, (255, 0, 0), 2)
    return out


def dibujar_solucion(foto: np.ndarray, M: np.ndarray, dado: Board, resuelto: Board,
                     size: int = 450) -> np.ndarray:
    """Dibuja los digitos resueltos (solo los que faltaban) sobre la foto original,
    proyectados con la perspectiva inversa M^-1."""
    overlay = np.zeros((size, size, 3), np.uint8)
    step = size // 9
    for r in range(9):
        for c in range(9):
            if dado[r][c] == 0 and resuelto[r][c] != 0:
                org = (c * step + step // 3, r * step + 2 * step // 3)
                cv2.putText(overlay, str(resuelto[r][c]), org, cv2.FONT_HERSHEY_SIMPLEX,
                            step / 55, (0, 170, 0), 2)
    h, w = foto.shape[:2]
    back = cv2.warpPerspective(overlay, np.linalg.inv(M), (w, h))
    mask = back.any(axis=2)
    out = foto.copy()
    out[mask] = back[mask]
    return out


class SudokuPipeline:
    """Orquesta los 4 pasos. Reutiliza modelos cargados (eficiente para la app)."""

    def __init__(self, device: str = "mps", size: int = 450,
                 detector_weights=detector.DEFAULT_WEIGHTS,
                 recognizer_weights=recognizer.DEFAULT_WEIGHTS):
        self.size = size
        self.detector = detector.SudokuDetector(weights=detector_weights, device=device)
        self.cropper = cropper.SudokuCropper(det=self.detector, size=size)
        self.recognizer = recognizer.DigitRecognizer(weights=recognizer_weights, device=device)

    def procesar(self, imagen: np.ndarray) -> Resultado:
        """Recorre todo el pipeline y devuelve un Resultado con cada etapa."""
        res = Resultado(imagen=imagen)

        # 1) Deteccion del tablero
        res.imagen_anotada, res.deteccion = self.detector.detect_and_draw(imagen)
        if res.deteccion is None:
            res.error = "No se detecto ningun tablero de sudoku en la foto."
            return res

        # 2) Recorte + rectificacion
        try:
            board, M = self.cropper.crop_board(imagen)
        except Exception as e:
            res.error = f"No se pudo rectificar el tablero: {e}"
            return res
        res.tablero, res.M = board, M

        # 3) Reconocimiento de digitos (la red neuronal)
        cells = cropper.split_cells(board, 9)
        res.leido = self.recognizer.recognize_grid(cells)
        res.tablero_leido = dibujar_lectura(board, res.leido, self.size)

        # 4) Verificacion + resolucion (programatica)
        res.valido = solver.is_valid_board(res.leido)
        if not res.valido:
            res.error = "La lectura tiene conflictos: algun digito se leyo mal."
            return res
        res.solucion = solver.solve(res.leido)
        if res.solucion is None:
            res.error = "El sudoku leido no tiene solucion (probable error de lectura)."
            return res
        res.imagen_solucion = dibujar_solucion(imagen, M, res.leido, res.solucion, self.size)
        return res


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/main.py <foto.jpg>")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("No se pudo leer la imagen:", sys.argv[1])
        raise SystemExit(1)
    res = SudokuPipeline().procesar(img)
    if res.leido:
        print("Leido por la red:")
        solver.print_board(res.leido)
    if res.solucion:
        print("\nSolucion:")
        solver.print_board(res.solucion)
        cv2.imwrite("solucion.png", res.imagen_solucion)
        print("\nGuardado solucion.png")
    else:
        print("\n", res.error)
