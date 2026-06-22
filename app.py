"""Visualizador Streamlit del pipeline de Sudoku.

Subes una foto y la app muestra, paso a paso y de forma visual:
  1. Deteccion del tablero (YOLO)
  2. Recorte y rectificacion
  3. Lectura de digitos por la red neuronal (CNN)
  4. Verificacion + resolucion programatica (solver, la "verdad de referencia")

Ejecutar:
    streamlit run app.py
"""

import sys
from pathlib import Path
import glob

import numpy as np
import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import main as pipeline_mod
import solver
import solver_gnn
import solver_cnn

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="wide")


# ============================================================ TEMA / ESTILO
def inyectar_estilo():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root{
            --bg:#0b0e16; --bg-2:#0f1320; --panel:#141a2a; --panel-2:#101626;
            --line:#1f2940; --line-2:#283450;
            --txt:#e7ecf5; --muted:#8b96ad;
            --cyan:#22d3ee; --blue:#3b82f6; --lime:#a3e635; --amber:#f59e0b;
            --grad:linear-gradient(90deg,var(--cyan),var(--blue));
        }

        /* fondo general */
        .stApp{
            background:
                radial-gradient(900px 500px at 12% -8%, rgba(34,211,238,.10) 0%, transparent 55%),
                radial-gradient(900px 500px at 108% 6%, rgba(59,130,246,.12) 0%, transparent 55%),
                var(--bg);
            color:var(--txt);
        }
        html, body, [class*="css"]{ font-family:'Space Grotesk', sans-serif; color:var(--txt); }
        h1,h2,h3,h4{ font-family:'Space Grotesk', sans-serif; color:var(--txt); letter-spacing:-.4px; }
        p, label, span, li{ color:var(--txt); }

        /* ---------- HERO ---------- */
        .hero{
            position:relative; overflow:hidden;
            border-radius:18px; padding:34px 36px;
            background:
                linear-gradient(180deg, rgba(34,211,238,.06), transparent 60%),
                var(--panel);
            border:1px solid var(--line);
            box-shadow:0 24px 60px -30px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.03);
            margin-bottom:14px;
        }
        /* línea de acento superior */
        .hero::before{
            content:""; position:absolute; top:0; left:0; right:0; height:3px;
            background:var(--grad);
        }
        /* rejilla técnica de fondo */
        .hero::after{
            content:""; position:absolute; inset:0; opacity:.5;
            background-image:
                linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
            background-size:26px 26px; pointer-events:none;
        }
        .hero h1{ font-size:2.35rem; margin:0; font-weight:700; position:relative; z-index:1; }
        .hero h1 .accent{
            background:var(--grad); -webkit-background-clip:text; background-clip:text;
            -webkit-text-fill-color:transparent;
        }
        .hero p{ color:var(--muted); font-size:1.02rem; margin:.7rem 0 0; max-width:720px; position:relative; z-index:1; }
        .hero .tag{
            display:inline-block; margin-bottom:14px; position:relative; z-index:1;
            font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:1.5px;
            text-transform:uppercase; color:var(--cyan);
            padding:5px 12px; border:1px solid var(--line-2); border-radius:6px;
            background:rgba(34,211,238,.06);
        }

        /* ---------- BADGES de paso ---------- */
        .step{
            display:flex; align-items:center; gap:14px;
            font-weight:600; font-size:1.18rem; color:var(--txt);
            padding:12px 4px 10px; margin:18px 0 10px;
            border-bottom:1px solid var(--line);
        }
        .step .n{
            display:inline-grid; place-items:center;
            width:34px; height:34px; border-radius:9px;
            font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.95rem;
            color:var(--bg); flex:0 0 auto;
        }
        .s1 .n{ background:var(--cyan); box-shadow:0 0 18px -2px rgba(34,211,238,.6); }
        .s2 .n{ background:var(--blue); box-shadow:0 0 18px -2px rgba(59,130,246,.6); }
        .s3 .n{ background:var(--lime); box-shadow:0 0 18px -2px rgba(163,230,53,.5); }
        .s4 .n{ background:var(--amber); box-shadow:0 0 18px -2px rgba(245,158,11,.5); }

        /* ---------- tarjetas / imágenes ---------- */
        [data-testid="stImage"]{
            background:var(--panel); border:1px solid var(--line);
            border-radius:14px; padding:8px; transition:border-color .2s;
        }
        [data-testid="stImage"]:hover{ border-color:var(--line-2); }
        [data-testid="stImage"] img{ border-radius:8px; }
        [data-testid="stImage"] figcaption{
            font-family:'JetBrains Mono', monospace; font-size:.78rem; letter-spacing:.5px;
            color:var(--muted); text-align:center; padding-top:8px;
        }

        /* métricas */
        [data-testid="stMetric"]{
            background:var(--panel); border:1px solid var(--line);
            border-radius:14px; padding:16px 18px;
            position:relative; overflow:hidden;
        }
        [data-testid="stMetric"]::before{
            content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--grad);
        }
        [data-testid="stMetricValue"]{ color:var(--cyan); font-weight:700; }
        [data-testid="stMetricLabel"]{ color:var(--muted); }

        /* alerts */
        [data-testid="stAlert"]{
            border-radius:12px; font-weight:500;
            background:var(--panel) !important; border:1px solid var(--line) !important;
            color:var(--txt) !important;
        }

        /* sidebar */
        [data-testid="stSidebar"]{
            background:var(--bg-2); border-right:1px solid var(--line);
        }
        [data-testid="stSidebar"] *{ color:var(--txt); }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
            background:var(--panel-2); border:1px dashed var(--line-2); border-radius:12px;
        }

        /* botones */
        .stButton>button, [data-testid="stFileUploader"] button{
            border-radius:10px !important; font-weight:600 !important;
            border:1px solid var(--line-2) !important;
            background:var(--panel) !important; color:var(--cyan) !important;
        }
        .stButton>button:hover, [data-testid="stFileUploader"] button:hover{
            border-color:var(--cyan) !important;
        }

        /* selectbox */
        [data-testid="stSidebar"] [data-baseweb="select"]>div{
            background:var(--panel-2); border-color:var(--line-2); border-radius:10px;
        }

        /* expander */
        [data-testid="stExpander"]{
            border-radius:12px; border:1px solid var(--line); overflow:hidden;
            background:var(--panel-2);
        }

        /* chip dispositivo */
        .chip{
            display:inline-block; padding:5px 12px; border-radius:8px;
            font-family:'JetBrains Mono', monospace; font-size:.78rem; letter-spacing:.5px;
            background:var(--panel); border:1px solid var(--line-2); color:var(--lime);
        }
        code{ color:var(--cyan); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- utilidades
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def paso(n, titulo, clase):
    st.markdown(
        f'<div class="step {clase}"><span class="n">{n}</span>{titulo}</div>',
        unsafe_allow_html=True,
    )


def grid_html(dado, resuelto=None):
    """Renderiza una rejilla 9x9 en HTML. Digitos dados en cian; si se pasa
    `resuelto`, los rellenados salen en lima."""
    css = """
    <style>
    .sk-wrap{display:flex;justify-content:center;padding:8px 0}
    .sk{border-collapse:collapse;font-family:'JetBrains Mono',monospace;
        border-radius:12px;overflow:hidden;background:#0f1320;
        border:1px solid #283450;
        box-shadow:0 24px 50px -28px rgba(0,0,0,.9), 0 0 0 1px rgba(34,211,238,.05)}
    .sk td{width:42px;height:42px;text-align:center;font-size:20px;
           border:1px solid #1c263d}
    .sk tr:nth-child(3n) td{border-bottom:2px solid #34507a}
    .sk td:nth-child(3n){border-right:2px solid #34507a}
    .sk tr:first-child td{border-top:2px solid #34507a}
    .sk td:first-child{border-left:2px solid #34507a}
    .given{color:#22d3ee;font-weight:700;background:rgba(34,211,238,.05)}
    .solved{color:#a3e635;font-weight:500;background:rgba(163,230,53,.04)}
    .empty{color:#2c3855}
    </style>
    """
    rows = ""
    for r in range(9):
        cells = ""
        for c in range(9):
            d = dado[r][c]
            if d:
                cells += f'<td class="given">{d}</td>'
            elif resuelto is not None and resuelto[r][c]:
                cells += f'<td class="solved">{resuelto[r][c]}</td>'
            else:
                cells += '<td class="empty">·</td>'
        rows += f"<tr>{cells}</tr>"
    return css + f'<div class="sk-wrap"><table class="sk">{rows}</table></div>'


@st.cache_resource(show_spinner="Cargando modelos (YOLO + CNN)...")
def cargar_pipeline():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline_mod.SudokuPipeline(device=device), device


@st.cache_resource(show_spinner="Cargando solver neuronal (GNN)...")
def cargar_gnn(device):
    """Carga la GNN entrenada. Devuelve None si todavia no hay modelo."""
    try:
        return solver_gnn.GNNSolver(device=device)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Cargando solver neuronal (CNN 1M)...")
def cargar_cnn():
    """Carga la CNN (Keras) entrenada en 1M. Devuelve None si no hay modelo."""
    try:
        return solver_cnn.CNNSolver()
    except FileNotFoundError:
        return None


# ============================================================ APP
inyectar_estilo()

st.markdown(
    """
    <div class="hero">
        <span class="tag">YOLO · CNN · Solver</span>
        <h1>Sudoku <span class="accent">Vision Pipeline</span></h1>
        <p>Detección con YOLO → recorte y rectificación → lectura con una red
        neuronal (CNN) → verificación y resolución con el solver. La red
        <b>lee</b> el tablero; el solver <b>comprueba y resuelve</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pipe, device = cargar_pipeline()

with st.sidebar:
    st.header("🎛️ Entrada")
    st.markdown(f'<span class="chip">⚙️ Dispositivo: {device}</span>', unsafe_allow_html=True)
    st.markdown(" ")
    subida = st.file_uploader("Sube una foto de un sudoku", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    ejemplos = sorted(glob.glob(str(ROOT / "data/images/test/*.jpg")))
    usar_ejemplo = st.selectbox("…o prueba con un ejemplo de test",
                                ["(ninguno)"] + [Path(p).name for p in ejemplos])

# ---------------------------------------------------------------- cargar imagen
imagen = None
if subida is not None:
    data = np.frombuffer(subida.read(), np.uint8)
    imagen = cv2.imdecode(data, cv2.IMREAD_COLOR)
elif usar_ejemplo != "(ninguno)":
    imagen = cv2.imread(str(ROOT / "data/images/test" / usar_ejemplo))

if imagen is None:
    st.info("👈 Sube una foto o elige un ejemplo en la barra lateral para empezar.")
    st.stop()

# ---------------------------------------------------------------- procesar
with st.spinner("✨ Procesando tu sudoku..."):
    res = pipe.procesar(imagen)

# ---------------------------------------------------------------- etapas
paso("1", "Detección y recorte del tablero", "s1")
c1, c2, c3 = st.columns(3)
st.divider()
with c1:
    st.image(bgr2rgb(res.imagen), caption="📷 Foto original", use_container_width=True)
with c2:
    if res.imagen_anotada is not None:
        conf = res.deteccion[4] if res.deteccion else 0
        st.image(bgr2rgb(res.imagen_anotada),
                 caption=f"🎯 Tablero detectado (confianza {conf:.0%})", use_container_width=True)
with c3:
    if res.tablero is not None:
        st.image(bgr2rgb(res.tablero), caption="📐 Tablero rectificado", use_container_width=True)

if res.deteccion is None:
    st.error(f"❌ {res.error}")
    st.stop()

st.markdown("")
paso("2", "Lectura de los dígitos (red neuronal)", "s2")
c1, c2 = st.columns([1, 1])
with c1:
    if res.tablero is not None:
        st.image(bgr2rgb(res.tablero),
                 caption="📐 Tablero rectificado", use_container_width=True)
with c2:
    st.markdown("**Matriz leída por la red:**")
    st.markdown(grid_html(res.leido), unsafe_allow_html=True)
    ocupadas = sum(1 for row in res.leido for v in row if v)
    st.metric("Casillas con dígito detectadas", ocupadas)

st.markdown("")
# La lectura debe ser un sudoku resoluble: hace falta tanto para comparar los
# modelos neuronales como para mostrar la solucion de referencia del solver.
if not res.valido:
    st.error(f"❌ {res.error}")
    st.warning("El solver no puede resolver una lectura con conflictos. "
               "Suele deberse a 1-2 dígitos mal leídos.")
    st.stop()
if res.solucion is None:
    st.error(f"❌ {res.error}")
    st.stop()
st.success("✅ La lectura es un sudoku válido (sin conflictos).")

# ---------------------------------------------------------------- modelos neuronales
st.markdown("")
paso("3", "Solución por modelos neuronales", "s3")
st.caption("Dos redes que **aprenden** a resolver, cada una comparada contra el "
           "**backtracking** de `solver.py` (la verdad de referencia, siempre exacta).")

gnn = cargar_gnn(device)
cnn = cargar_cnn()

# calcular la solucion de cada modelo (None si el modelo no esta entrenado)
sol_gnn = gnn.solve(res.leido) if gnn is not None else None
with st.spinner("La CNN resuelve iterando celda a celda..."):
    sol_cnn = cnn.solve(res.leido) if cnn is not None else None


def bloque_modelo(nombre, descripcion, solucion):
    """Pinta la rejilla de un modelo y sus metricas frente al solver."""
    if solucion is None:
        st.info(f"ℹ️ Modelo **{nombre}** no disponible todavía.")
        return None
    met = solver.evaluar_solucion(solucion, res.solucion)
    st.markdown(f"**{nombre}** &nbsp;<span style='color:#8b96ad;font-size:.85rem'>"
                f"{descripcion}</span>", unsafe_allow_html=True)
    st.markdown(grid_html(res.leido, solucion), unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("Celdas correctas", f"{met['celdas_correctas']}/81")
    b.metric("Precisión", f"{met['precision']:.0%}")
    if met["exacta"]:
        st.success("✅ Tablero **exacto** (idéntico al solver).")
    else:
        st.warning("⚠️ No coincide del todo con el solver.")
    return met

cgnn, ccnn = st.columns(2)
with cgnn:
    met_gnn = bloque_modelo("GNN (red de grafos)",
                            "paso de mensajes entre celdas", sol_gnn)
with ccnn:
    met_cnn = bloque_modelo("CNN (entrenada en 1M)",
                            "iterativa, rellena la celda más segura", sol_cnn)

# ---------------------------------------------------------------- solver de referencia
st.markdown("")
paso("4", "Verificación y resolución (programática)", "s4")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(
        "**Solución** &nbsp; <span style='font-family:JetBrains Mono;font-size:.8rem;color:#8b96ad'>"
        "<span style='color:#22d3ee'>■</span> leído &nbsp; "
        "<span style='color:#a3e635'>■</span> resuelto</span>",
        unsafe_allow_html=True,
    )
    st.markdown(grid_html(res.leido, res.solucion), unsafe_allow_html=True)
with c2:
    if res.imagen_solucion is not None:
        st.image(bgr2rgb(res.imagen_solucion),
                 caption="🖼️ Solución superpuesta sobre la foto", use_container_width=True)

st.success("🎉 ¡Sudoku resuelto!")

# ---- tabla resumen: los tres motores frente a la referencia ----
st.markdown("**Comparativa de los tres motores de resolución**")
filas = [("🔢 Backtracking (solver.py)", "81/81", "100%", "✅ (referencia)")]
if met_gnn is not None:
    filas.append(("🕸️ GNN", f"{met_gnn['celdas_correctas']}/81",
                  f"{met_gnn['precision']:.0%}", "✅" if met_gnn["exacta"] else "❌"))
if met_cnn is not None:
    filas.append(("🧠 CNN (1M)", f"{met_cnn['celdas_correctas']}/81",
                  f"{met_cnn['precision']:.0%}", "✅" if met_cnn["exacta"] else "❌"))
st.table({
    "Motor": [f[0] for f in filas],
    "Celdas correctas": [f[1] for f in filas],
    "Precisión": [f[2] for f in filas],
    "Exacta": [f[3] for f in filas],
})
st.caption("El **backtracking** es el motor fiable; GNN y CNN son alternativas "
           "**aprendidas** que se miden contra él. Ambas parten de lo que leyó la CNN "
           "de dígitos, así que un error de lectura las afecta igual.")
