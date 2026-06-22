# 🧩 Sudoku Solver — Vision Pipeline

De una **foto** de un sudoku a su **solución**, paso a paso. El proyecto combina
visión por computador y deep learning en un pipeline completo: detecta el tablero,
lo recorta y rectifica, lee los dígitos con una red neuronal y resuelve el puzzle.
Además incluye **tres motores de resolución** (uno clásico y dos aprendidos) que se
comparan entre sí.

Todo se visualiza en una app de **Streamlit**.

---

## ✨ ¿Qué hace?

El pipeline encadena 4 etapas (`src/main.py`):

| # | Etapa | Cómo | Código |
|---|-------|------|--------|
| 1 | **Detección** del tablero en la foto | YOLOv8 (Ultralytics) | [src/detector.py](src/detector.py) |
| 2 | **Recorte y rectificación** a vista cuadrada 450×450 | esquinas + `warpPerspective` (OpenCV) | [src/cropper.py](src/cropper.py) |
| 3 | **Lectura** de los 81 dígitos | CNN entrenada en MNIST (PyTorch) | [src/recognizer.py](src/recognizer.py) |
| 4 | **Verificación y resolución** | backtracking exacto | [src/solver.py](src/solver.py) |

### Tres motores de resolución

La lectura de la red se resuelve con tres motores distintos, todos medidos contra
la misma referencia:

- 🔢 **Backtracking** ([src/solver.py](src/solver.py)) — algorítmico, **siempre exacto**. Es la *verdad de referencia*.
- 🕸️ **GNN** ([src/solver_gnn.py](src/solver_gnn.py)) — red de grafos (paso de mensajes entre celdas, estilo *Recurrent Relational Network*). Las 81 celdas son nodos conectados a sus 20 *peers* (fila, columna y bloque).
- 🧠 **CNN 1M** ([src/solver_cnn.py](src/solver_cnn.py)) — CNN (Keras) entrenada en 1M de sudokus; resuelve iterativamente rellenando la celda vacía más segura.

El backtracking sirve de oráculo: `solver.evaluar_solucion` mide cuántas celdas
aciertan la GNN y la CNN frente a la solución exacta.

---

## 🚀 Uso

### App de Streamlit (recomendado)

```bash
streamlit run app.py
```

Sube una foto de un sudoku (o elige un ejemplo de `data/images/test/`) y la app
muestra cada etapa del pipeline de forma visual, además de la comparativa de los
tres motores.

### Línea de comandos

```bash
python src/main.py foto.jpg        # imprime la solución y guarda solucion.png
```

Cada módulo de `src/` también es ejecutable de forma independiente para depurar una
etapa concreta (ver el bloque `__main__` de cada archivo), por ejemplo:

```bash
python src/detector.py foto.jpg    # guarda detected.png con la caja del tablero
```

---

## 📦 Instalación

Requiere **Python 3.12**.

```bash
# entorno virtual
python -m venv .venv
source .venv/bin/activate

# dependencias
pip install -r requirements.txt
```

> En Mac con Apple Silicon, el pipeline usa **MPS** automáticamente si está
> disponible; si no, cae a CPU.

---

## 🧠 Modelos

Los pesos entrenados están en la carpeta `models/`:

| Archivo | Modelo | Se entrena en |
|---------|--------|---------------|
| `models/sudoku_detector.pt` | YOLO (detección del tablero) | [notebooks/1_detector_dev.ipynb](notebooks/1_detector_dev.ipynb) |
| `models/digit_cnn.pt` | CNN de dígitos (MNIST) | [notebooks/3_recognizer_dev.ipynb](notebooks/3_recognizer_dev.ipynb) |
| `models/sudoku_solver_gnn.pt` | GNN solver | [notebooks/6_solver_gnn_dev.ipynb](notebooks/6_solver_gnn_dev.ipynb) |
| `models/cnn_1m.keras` | CNN solver (1M sudokus) | [notebooks/5_solver_nn_dev.ipynb](notebooks/5_solver_nn_dev.ipynb) |

La app degrada con elegancia: si falta algún modelo de solver (GNN/CNN), lo marca
como *no disponible* y sigue funcionando con el backtracking.

---

## 📓 Notebooks de desarrollo

Cada etapa se desarrolla y entrena en su propio notebook (`notebooks/`):

1. [`1_detector_dev.ipynb`](notebooks/1_detector_dev.ipynb) — detección con YOLO
2. [`2_cropper_dev.ipynb`](notebooks/2_cropper_dev.ipynb) — recorte y rectificación
3. [`3_recognizer_dev.ipynb`](notebooks/3_recognizer_dev.ipynb) — lectura de dígitos (CNN/MNIST)
4. [`4_solver_dev.ipynb`](notebooks/4_solver_dev.ipynb) — solver por backtracking
5. [`5_solver_nn_dev.ipynb`](notebooks/5_solver_nn_dev.ipynb) — CNN solver (1M)
6. [`6_solver_gnn_dev.ipynb`](notebooks/6_solver_gnn_dev.ipynb) — GNN solver

---

## 🗂️ Estructura

```
.
├── app.py                 # app de Streamlit (visualizador del pipeline)
├── src/
│   ├── main.py            # orquesta las 4 etapas (SudokuPipeline)
│   ├── detector.py        # YOLO: detección del tablero
│   ├── cropper.py         # recorte + rectificación + split en 81 casillas
│   ├── recognizer.py      # CNN: lectura de dígitos
│   ├── solver.py          # backtracking (referencia exacta)
│   ├── solver_gnn.py      # solver GNN (red de grafos)
│   └── solver_cnn.py      # solver CNN (1M sudokus, Keras)
├── notebooks/             # desarrollo y entrenamiento de cada etapa
├── data/
│   ├── images/{train,val,test}/   # dataset de tableros
│   ├── labels/{train,val,test}/   # etiquetas YOLO
│   └── sudoku.yaml                # config del dataset YOLO
└── models/                # pesos entrenados
```

---

## 🛠️ Stack

OpenCV · PyTorch · Ultralytics (YOLOv8) · Keras · Streamlit · NumPy
