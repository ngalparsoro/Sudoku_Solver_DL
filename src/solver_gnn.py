"""Solver neuronal por red de grafos (GNN, estilo Recurrent Relational Network).

A diferencia de `solver.py` (backtracking exacto, siempre correcto), esta red
APRENDE a resolver: las 81 celdas son nodos, cada uno conectado a las 20 con las
que comparte fila, columna o bloque, y se intercambian mensajes durante varios
pasos. Se entrena en `notebooks/6_solver_gnn_dev.ipynb`.

Expone:
  - SudokuGNN: la red (misma arquitectura que el notebook).
  - GNNSolver: carga los pesos y resuelve un tablero 9x9.
"""

from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "sudoku_solver_gnn.pt"

Board = list[list[int]]


def construir_vecinos() -> torch.Tensor:
    """Para cada celda, las 20 'peers' (misma fila, columna o bloque). (81, 20)."""
    vecinos = []
    for r in range(9):
        for c in range(9):
            peers = set()
            for cc in range(9):
                peers.add(r * 9 + cc)               # fila
            for rr in range(9):
                peers.add(rr * 9 + c)               # columna
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for rr in range(br, br + 3):             # bloque 3x3
                for cc in range(bc, bc + 3):
                    peers.add(rr * 9 + cc)
            peers.discard(r * 9 + c)
            vecinos.append(sorted(peers))
    return torch.tensor(vecinos, dtype=torch.long)   # (81, 20)


class SudokuGNN(nn.Module):
    """Paso de mensajes recurrente sobre el grafo del sudoku."""

    def __init__(self, vecinos: torch.Tensor, H: int = 96, steps: int = 8):
        super().__init__()
        self.register_buffer("nb", vecinos)          # (81, 20)
        self.H, self.steps = H, steps
        self.embed = nn.Embedding(10, H)             # digito 0-9 -> vector H
        self.msg = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(),
                                 nn.Linear(H, H), nn.ReLU())
        self.gru = nn.GRUCell(2 * H, H)              # entrada: [mensaje, embedding inicial]
        self.out = nn.Linear(H, 9)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.size(0)
        xe = self.embed(x)                           # (B,81,H) rasgo fijo de entrada
        h = xe.clone()                               # estado inicial de cada nodo
        salidas = []
        for _ in range(self.steps):
            hj = h[:, self.nb]                                   # (B,81,20,H) vecinos
            hi = h.unsqueeze(2).expand(-1, -1, self.nb.size(1), -1)
            m = self.msg(torch.cat([hi, hj], -1)).sum(2)         # (B,81,H) mensajes sumados
            gin = torch.cat([m, xe], -1).reshape(B * 81, -1)
            h = self.gru(gin, h.reshape(B * 81, self.H)).reshape(B, 81, self.H)
            salidas.append(self.out(h))              # (B,81,9) prediccion de este paso
        return salidas


class GNNSolver:
    """Carga la GNN entrenada y resuelve tableros 9x9."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, device: str = "cpu",
                 H: int = 96, steps: int = 8):
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No existe el modelo {weights}. Entrenalo antes "
                "(ver notebooks/6_solver_gnn_dev.ipynb)."
            )
        self.device = device
        self.model = SudokuGNN(construir_vecinos(), H=H, steps=steps).to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def solve(self, board: Board) -> Board:
        """Devuelve el tablero 9x9 que predice la red (siempre lleno, 1-9)."""
        flat = [board[r][c] for r in range(9) for c in range(9)]
        x = torch.tensor(flat, dtype=torch.long, device=self.device).view(1, 81)
        pred = self.model(x)[-1].argmax(-1).cpu().numpy().reshape(9, 9) + 1
        return pred.tolist()


if __name__ == "__main__":
    import solver

    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    gnn = GNNSolver()
    pred = gnn.solve(puzzle)
    ref = solver.solve(puzzle)
    print("GNN:"); solver.print_board(pred)
    print("\nMetricas vs solver:", solver.evaluar_solucion(pred, ref))
