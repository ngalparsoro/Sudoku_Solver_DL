"""Solver neuronal por CNN (Keras) entrenada en 1M de sudokus.

Entrada: tablero 9x9 aplanado (81 valores 0-9, sin normalizar).
Salida del modelo: (81, 10) -> por cada celda, logits de los digitos 0-9.

Se resuelve ITERATIVAMENTE (como los CNN-solver clasicos del dataset de 1M):
en cada paso se rellena solo la celda vacia con mayor confianza y se vuelve a
predecir, hasta completar el tablero. Una sola pasada acierta pocas celdas; iterar
mejora bastante, aunque sigue sin ser tan fiable como el backtracking de solver.py.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "cnn_1m.keras"

Board = list[list[int]]


class CNNSolver:
    """Carga la CNN entrenada (Keras) y resuelve tableros 9x9."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS):
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No existe el modelo {weights}. Entrenalo antes (CNN de 1M sudokus)."
            )
        import keras
        self.model = keras.models.load_model(str(weights))

    def _probs(self, flat: np.ndarray) -> np.ndarray:
        """(81,) enteros -> (81,10) probabilidades (softmax)."""
        out = np.asarray(self.model(flat[None], training=False))[0]   # (81,10)
        e = np.exp(out - out.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def solve(self, board: Board, iterativo: bool = True) -> Board:
        """Devuelve el tablero 9x9 que predice la red.

        iterativo=True rellena de una en una la celda vacia mas segura (mejor);
        iterativo=False hace una sola pasada (mas rapido, menos preciso).
        """
        grid = np.array(board, np.float32).reshape(81)
        if not iterativo:
            return self._probs(grid).argmax(1).reshape(9, 9).astype(int).tolist()
        grid = grid.copy()
        while (grid == 0).any():
            p = self._probs(grid)
            libres = grid == 0
            i = np.where(libres, p.max(1), -1.0).argmax()    # celda vacia mas segura
            grid[i] = p[i].argmax()
        return grid.reshape(9, 9).astype(int).tolist()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import solver

    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    cnn = CNNSolver()
    pred = cnn.solve(puzzle)
    ref = solver.solve(puzzle)
    print("CNN (iterativo):"); solver.print_board(pred)
    print("\nMetricas vs solver:", solver.evaluar_solucion(pred, ref))
