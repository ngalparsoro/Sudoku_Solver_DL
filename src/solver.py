"""Solver de Sudoku por backtracking (solucion algoritmica, siempre correcta).

Doble proposito:
  1. Resolver el sudoku leido de la foto (cierra la cadena foto -> solucion).
  2. Servir de "verdad de referencia" (oraculo) para, cuando exista un modelo
     generador de soluciones, medir su eficiencia/precision con `evaluar_solucion`.

El tablero es una lista de 9 listas (9x9) de enteros; 0 = casilla vacia.
La lista plana de 81 que devuelve el recognizer se convierte con `desde_lista`.
"""

from copy import deepcopy

Board = list[list[int]]


def desde_lista(flat: list[int]) -> Board:
    """Convierte la lista plana de 81 (la del recognizer) en una matriz 9x9."""
    if len(flat) != 81:
        raise ValueError(f"Se esperaban 81 valores, hay {len(flat)}")
    return [flat[i * 9:(i + 1) * 9] for i in range(9)]


def find_empty(board: Board) -> tuple[int, int] | None:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def is_valid(board: Board, row: int, col: int, num: int) -> bool:
    """True si colocar `num` en (row, col) respeta fila, columna y bloque 3x3."""
    if any(board[row][c] == num for c in range(9)):
        return False
    if any(board[r][col] == num for r in range(9)):
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == num:
                return False
    return True


def is_valid_board(board: Board) -> bool:
    """Comprueba que el tablero de partida no tenga conflictos (util tras la lectura)."""
    if len(board) != 9 or any(len(row) != 9 for row in board):
        return False
    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n == 0:
                continue
            if not (1 <= n <= 9):
                return False
            board[r][c] = 0
            ok = is_valid(board, r, c, n)
            board[r][c] = n
            if not ok:
                return False
    return True


def _solve(board: Board) -> bool:
    empty = find_empty(board)
    if empty is None:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if _solve(board):
                return True
            board[row][col] = 0   # retroceso
    return False


def solve(board: Board) -> Board | None:
    """Resuelve sin mutar la entrada. Devuelve la solucion 9x9 o None si no tiene."""
    if not is_valid_board(board):
        return None
    work = deepcopy(board)
    return work if _solve(work) else None


def count_solutions(board: Board, limit: int = 2) -> int:
    """Cuenta soluciones hasta `limit` (para saber si el puzzle es unico: ==1)."""
    work = deepcopy(board)
    total = 0

    def bt() -> None:
        nonlocal total
        if total >= limit:
            return
        empty = find_empty(work)
        if empty is None:
            total += 1
            return
        r, c = empty
        for n in range(1, 10):
            if is_valid(work, r, c, n):
                work[r][c] = n
                bt()
                work[r][c] = 0

    bt()
    return total


def is_complete_solution(board: Board) -> bool:
    """True si el tablero esta completo y es valido (1-9 en cada fila/col/bloque)."""
    full = set(range(1, 10))
    for i in range(9):
        if {board[i][c] for c in range(9)} != full:
            return False
        if {board[r][i] for r in range(9)} != full:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if {board[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)} != full:
                return False
    return True


def evaluar_solucion(predicha: Board, referencia: Board) -> dict:
    """Compara la solucion de un modelo (`predicha`) contra la del solver (`referencia`).

    Pensado para medir la eficiencia de un futuro modelo generador de soluciones.
    Devuelve metricas: celdas correctas, precision y si es exacta.
    """
    correctas = sum(predicha[r][c] == referencia[r][c] for r in range(9) for c in range(9))
    return {
        "celdas_correctas": correctas,
        "total": 81,
        "precision": correctas / 81,
        "exacta": correctas == 81,
        "solucion_valida": is_complete_solution(predicha),
    }


def print_board(board: Board) -> None:
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("------+-------+------")
        row = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row += "| "
            row += (str(board[r][c]) if board[r][c] else ".") + " "
        print(row.rstrip())


if __name__ == "__main__":
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    print("Puzzle:")
    print_board(puzzle)
    sol = solve(puzzle)
    print("\nSolucion:")
    print_board(sol) if sol else print("sin solucion")
    print("\nunica?:", count_solutions(puzzle) == 1)
