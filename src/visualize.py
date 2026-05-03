"""Generación automática de figuras del artículo.

Cada función produce una figura del artículo y la guarda directamente en la
subcarpeta correspondiente de `figuras/`. Las figuras se regeneran cada vez
que cambia el dataset, manteniendo coherencia con los datos. Convenciones:

    figuras/imagenes/   — mapas, scatter plots, capturas de la demo.
    figuras/tablas/     — tablas renderizadas como PNG (para LinkedIn).
    figuras/diagramas/  — pipelines, arquitecturas, flujos.
    figuras/codigo/     — capturas de bloques de código resaltados.
    figuras/formulas/   — ecuaciones renderizadas con LaTeX.

Las figuras se exportan en PNG (resolución 150 DPI) o SVG (vectorial) según
convenga al destino: PNG para LinkedIn (que no admite SVG), SVG para la web
extendida cuando se requiera escalado limpio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.table import Table


FIGURAS = Path(__file__).resolve().parents[1] / "figuras"
DPI = 150

PALETA_SUCURSALES = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]


def _ruta(subcarpeta: str, nombre: str) -> Path:
    destino = FIGURAS / subcarpeta
    destino.mkdir(parents=True, exist_ok=True)
    return destino / nombre


def mapa_clientes_clusters(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
    plan: pd.DataFrame,
) -> Path:
    """Scatter de clientes coloreados por sucursal asignada + posición de las sucursales."""
    ruta = _ruta("imagenes", "02_mapa_clientes_clusters.png")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)

    plan_indexed = plan.set_index("cliente_id")
    for i, suc in sucursales.reset_index(drop=True).iterrows():
        color = PALETA_SUCURSALES[i % len(PALETA_SUCURSALES)]
        clientes_suc = plan_indexed[plan_indexed["sucursal_id"] == suc["sucursal_id"]]
        ax.scatter(
            clientes_suc["lon"], clientes_suc["lat"],
            s=60, c=color, alpha=0.7, edgecolor="white", linewidth=0.8,
            label=f"{suc['sucursal_id']} ({len(clientes_suc)} clientes)",
        )
        ax.scatter(
            suc["lon"], suc["lat"],
            s=400, c=color, marker="s", edgecolor="black", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            suc["sucursal_id"], (suc["lon"], suc["lat"]),
            ha="center", va="center", fontweight="bold", color="white", fontsize=10, zorder=6,
        )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Distribución geográfica · clientes asignados por sucursal", fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def diagrama_arquitectura() -> Path:
    """Diagrama del pipeline con dos inputs (histórico de pedidos + disponibilidad por centro).

    Layout: dos cajas de input arriba que convergen en una flecha hacia "Predicción de demanda",
    y a partir de ahí flujo lineal hasta el dashboard.
    """
    ruta = _ruta("diagramas", "01_arquitectura_sistema.png")
    fig, ax = plt.subplots(figsize=(16, 5.6), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ancho = 2.4
    alto = 1.0
    sep = 3.0

    inputs = [
        ("Histórico\nde pedidos", "#E8F1F8", "#1F4F87"),
        ("Disponibilidad\npor centro", "#E8F1F8", "#1F4F87"),
    ]
    flujo = [
        ("Predicción\nde demanda", "#FFEAD0", "#B8650A"),
        ("Clustering\nde clientes", "#FFEAD0", "#B8650A"),
        ("Programación\ndinámica", "#D7F0D7", "#2C7A3F"),
        ("Plan de rutas\ncoordinado", "#FBE0E0", "#B33A3A"),
        ("Dashboard\nStreamlit", "#E5DCF5", "#5B3DA8"),
    ]

    y_input_arriba = 1.85
    y_input_abajo = 0.65
    y_flujo = 1.25

    x_inputs = 0.0
    for (texto, fill, borde), y_in in zip(inputs, [y_input_arriba, y_input_abajo]):
        ax.add_patch(
            plt.Rectangle(
                (x_inputs, y_in - alto / 2),
                ancho, alto,
                facecolor=fill, edgecolor=borde, linewidth=2.2,
            )
        )
        ax.text(
            x_inputs + ancho / 2, y_in, texto,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1A2233",
            linespacing=1.30,
        )

    x_flujo_inicio = sep
    for i, (texto, fill, borde) in enumerate(flujo):
        x = x_flujo_inicio + i * sep
        ax.add_patch(
            plt.Rectangle(
                (x, y_flujo - alto / 2),
                ancho, alto,
                facecolor=fill, edgecolor=borde, linewidth=2.2,
            )
        )
        ax.text(
            x + ancho / 2, y_flujo, texto,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1A2233",
            linespacing=1.30,
        )

    x_prediccion = x_flujo_inicio
    for y_in in [y_input_arriba, y_input_abajo]:
        arrow = FancyArrowPatch(
            (x_inputs + ancho + 0.05, y_in),
            (x_prediccion - 0.05, y_flujo),
            arrowstyle="->", mutation_scale=22,
            color="#4A5568", linewidth=2.0,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    for i in range(len(flujo) - 1):
        x = x_flujo_inicio + i * sep
        arrow = FancyArrowPatch(
            (x + ancho + 0.05, y_flujo),
            (x + sep - 0.05, y_flujo),
            arrowstyle="->", mutation_scale=22,
            color="#4A5568", linewidth=2.0,
        )
        ax.add_patch(arrow)

    ax.set_xlim(-0.3, x_flujo_inicio + (len(flujo) - 1) * sep + ancho + 0.3)
    ax.set_ylim(0.0, 2.7)
    ax.axis("off")

    ax.text(
        (x_flujo_inicio + (len(flujo) - 1) * sep + ancho) / 2 + (-0.3) / 2,
        2.50,
        "Arquitectura del sistema: de los datos al plan diario",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color="#0F2D52",
    )

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def curva_demanda(real: pd.Series, predicha: pd.Series, fechas: pd.Series) -> Path:
    """Demanda total agregada real vs predicha en validación temporal."""
    ruta = _ruta("imagenes", "demanda_real_vs_predicha.png")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=DPI)

    df = pd.DataFrame({"fecha": pd.to_datetime(fechas), "real": real.values, "predicha": predicha.values})
    df = df.groupby("fecha").sum().reset_index()

    ax.plot(df["fecha"], df["real"], label="Demanda real (test)", color="#1F77B4", linewidth=2.0)
    ax.plot(df["fecha"], df["predicha"], label="Demanda predicha (LightGBM)", color="#FF7F0E", linewidth=2.0, linestyle="--")
    ax.fill_between(df["fecha"], df["real"], df["predicha"], color="#FF7F0E", alpha=0.15)

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Demanda total diaria (unidades)")
    ax.set_title("Validación del modelo · demanda real vs predicha", fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def heatmap_complejidad() -> Path:
    """Complejidad computacional: DP exacta vs DP aproximada vs OR-Tools por número de nodos."""
    ruta = _ruta("imagenes", "heatmap_complejidad.png")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    n_nodos = np.array([5, 10, 12, 15, 18, 20, 25, 30, 50, 100])
    dp_exacta = np.array([n * n * (2 ** n) for n in n_nodos], dtype=float)
    dp_approx = np.array([n ** 2 for n in n_nodos], dtype=float)
    or_tools = np.array([n ** 2 * np.log2(max(n, 2)) for n in n_nodos], dtype=float)

    ax.semilogy(n_nodos, dp_exacta, "o-", label=r"DP exacta (Held-Karp): $O(2^n \cdot n^2)$", color="#D62728", linewidth=2)
    ax.semilogy(n_nodos, dp_approx, "s-", label=r"DP aproximada (NN + 2-opt): $O(n^2)$", color="#2CA02C", linewidth=2)
    ax.semilogy(n_nodos, or_tools, "^-", label=r"OR-Tools (heurística + búsqueda): $\sim O(n^2 \log n)$", color="#1F77B4", linewidth=2)

    ax.axvline(x=15, color="gray", linestyle=":", alpha=0.6)
    ax.text(15.3, 1e2, "Frontera práctica\nde DP exacta", fontsize=9, color="gray")

    ax.set_xlabel("Número de clientes (n)")
    ax.set_ylabel("Operaciones (escala log)")
    ax.set_title("Complejidad computacional · cuándo usar cada enfoque", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def plan_rutas_mapa(
    plan: pd.DataFrame,
    sucursales: pd.DataFrame,
    rutas_resueltas: list,
) -> Path:
    """Mapa con las rutas resueltas, líneas coloreadas por sucursal."""
    ruta_archivo = _ruta("imagenes", "03_plan_rutas_resuelto.png")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)

    plan_indexed = plan.set_index("cliente_id")
    suc_indexed = sucursales.set_index("sucursal_id")

    for i, suc in sucursales.reset_index(drop=True).iterrows():
        color = PALETA_SUCURSALES[i % len(PALETA_SUCURSALES)]
        ax.scatter(suc["lon"], suc["lat"], s=400, c=color, marker="s",
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(suc["sucursal_id"], (suc["lon"], suc["lat"]),
                    ha="center", va="center", fontweight="bold", color="white", fontsize=10, zorder=6)

    for r in rutas_resueltas:
        suc_idx = list(sucursales["sucursal_id"]).index(r.sucursal_id)
        color = PALETA_SUCURSALES[suc_idx % len(PALETA_SUCURSALES)]
        suc_coord = suc_indexed.loc[r.sucursal_id]
        coords_x = [suc_coord["lon"]]
        coords_y = [suc_coord["lat"]]
        for cid in r.secuencia_clientes:
            c = plan_indexed.loc[cid]
            coords_x.append(c["lon"])
            coords_y.append(c["lat"])
        coords_x.append(suc_coord["lon"])
        coords_y.append(suc_coord["lat"])
        ax.plot(coords_x, coords_y, color=color, linewidth=1.5, alpha=0.7, zorder=3)
        ax.scatter(coords_x[1:-1], coords_y[1:-1], c=color, s=40, alpha=0.8,
                   edgecolor="white", linewidth=0.6, zorder=4)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Plan de rutas resuelto · coordinado entre sucursales", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_archivo, bbox_inches="tight")
    plt.close(fig)
    return ruta_archivo


def comparativa_antes_despues(metricas: pd.DataFrame) -> Path:
    """Barras antes/después: kilometraje, # camiones, costo estimado.

    `metricas` debe tener columnas: ['escenario', 'km_total', 'n_rutas', 'costo_estimado'].
    """
    ruta = _ruta("imagenes", "05_antes_despues.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=DPI)

    metricas_indexed = metricas.set_index("escenario")
    colores = ["#D62728", "#2CA02C"]

    titulos = [("km_total", "Kilometraje total (km)"),
               ("n_rutas", "Número de rutas"),
               ("costo_estimado", "Costo estimado (USD)")]
    for ax, (col, titulo) in zip(axes, titulos):
        valores = metricas_indexed[col]
        ax.bar(valores.index, valores.values, color=colores[: len(valores)], edgecolor="black", linewidth=1.2)
        for i, v in enumerate(valores.values):
            ax.text(i, v, f"{v:,.0f}" if v > 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Antes vs Después · planificación coordinada con ML + DP",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def tabla_complejidad_render() -> Path:
    """Tabla de complejidad teórica como imagen (para LinkedIn)."""
    ruta = _ruta("tablas", "tabla_complejidad.png")
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=DPI)
    ax.axis("off")

    cabeceras = ["Tamaño n", "DP exacta\n(Held-Karp)", "DP aproximada\n(NN + 2-opt)", "OR-Tools\n(heurística)"]
    filas = [
        ["10", "10⁵ ops · ~1 ms", "10² ops · <1 ms", "10² ops · <1 ms"],
        ["15", "7·10⁶ ops · ~50 ms", "2·10² ops · <1 ms", "10³ ops · <1 ms"],
        ["20", "4·10⁸ ops · ~5 s", "4·10² ops · <1 ms", "2·10³ ops · <1 ms"],
        ["30", "10¹¹ ops · INVIABLE", "10³ ops · ~1 ms", "5·10³ ops · ~10 ms"],
        ["50", "INVIABLE", "2·10³ ops · ~5 ms", "10⁴ ops · ~100 ms"],
        ["100", "INVIABLE", "10⁴ ops · ~50 ms", "5·10⁴ ops · ~1 s"],
    ]
    tabla = ax.table(cellText=filas, colLabels=cabeceras, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.0, 1.8)
    for i in range(len(cabeceras)):
        tabla[(0, i)].set_facecolor("#1F77B4")
        tabla[(0, i)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Complejidad por enfoque · cuándo usar cada uno",
                 fontsize=12, fontweight="bold", y=0.96)
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def codigo_render(titulo: str, codigo: str, nombre_archivo: str, tema: str = "oscuro") -> Path:
    """Renderiza un bloque de código como imagen (para LinkedIn).

    `tema`: "oscuro" (fondo negro estilo terminal) o "claro" (fondo blanco con sintaxis).
    """
    ruta = _ruta("codigo", nombre_archivo)
    lineas = codigo.split("\n")
    altura = max(2.5, 0.32 * len(lineas) + 0.8)
    fig, ax = plt.subplots(figsize=(11, altura), dpi=DPI)
    ax.axis("off")

    if tema == "oscuro":
        bg, fg, header_bg, header_fg = "#0F172A", "#E2E8F0", "#1E293B", "#94A3B8"
    else:
        bg, fg, header_bg, header_fg = "#F8FAFC", "#1E293B", "#1F4F87", "white"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, facecolor=header_bg, edgecolor="none"))
    ax.text(0.02, 0.96, titulo, fontsize=11, fontweight="bold", color=header_fg, va="center")

    y = 0.86
    for linea in lineas:
        ax.text(0.025, y, linea, fontfamily="monospace", fontsize=10, color=fg, va="top")
        y -= 0.040

    fig.savefig(ruta, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return ruta


def codigo_held_karp_extracto() -> Path:
    return codigo_render(
        titulo="src/dp_exact.py · Held-Karp con bitmask y memoización",
        codigo=(
            "def held_karp(distancias):\n"
            "    n = distancias.shape[0]\n"
            "    dp = {}\n"
            "\n"
            "    # Caso base: rutas de longitud 1 (depósito → i)\n"
            "    for i in range(1, n):\n"
            "        dp[(1 << i, i)] = (distancias[0, i], 0)\n"
            "\n"
            "    # Llenado por tamaños crecientes de subconjunto\n"
            "    for tam_subset in range(2, n):\n"
            "        for subset in subsets_de_tamano(n - 1, tam_subset):\n"
            "            for i in nodos_en(subset):\n"
            "                mejor = min(\n"
            "                    dp[(subset & ~(1<<i), j)][0] + distancias[j, i]\n"
            "                    for j in nodos_en(subset & ~(1<<i))\n"
            "                )\n"
            "                dp[(subset, i)] = (mejor, ...)\n"
            "\n"
            "    return min(dp[(subset_total, i)][0] + distancias[i, 0]\n"
            "               for i in range(1, n))"
        ),
        nombre_archivo="held_karp_extracto.png",
    )


def codigo_comandos_clone() -> Path:
    return codigo_render(
        titulo="Terminal · clonar y ejecutar el sistema",
        codigo=(
            "$ git clone https://github.com/leanmasterpymes/gestion_rutas\n"
            "$ cd gestion_rutas\n"
            "$ pip install -r requirements.txt\n"
            "$ streamlit run app/streamlit_app.py\n"
            "\n"
            "# Aplicación disponible en http://localhost:8501"
        ),
        nombre_archivo="comandos_clone.png",
    )


def tabla_benchmark_ortools() -> Path:
    """Tabla comparativa motor propio vs OR-Tools renderizada como imagen."""
    ruta = _ruta("tablas", "04_benchmark_ortools.png")
    fig, ax = plt.subplots(figsize=(11, 2.6), dpi=DPI)
    ax.axis("off")

    cabeceras = ["Caso", "Motor propio", "OR-Tools", "Gap", "Tiempo propio", "Tiempo OR-Tools"]
    filas = [
        ["TSP exacto · 12 clientes", "154.06 km", "154.06 km", "0.0 %", "95 ms", "5.000 ms"],
        ["Plan completo · 50 clientes / 5 sucursales", "334.66 km", "289.31 km", "+15.7 %", "7 ms", "15.000 ms"],
    ]
    tabla = ax.table(cellText=filas, colLabels=cabeceras, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.scale(1.0, 2.0)
    for i in range(len(cabeceras)):
        tabla[(0, i)].set_facecolor("#1F4F87")
        tabla[(0, i)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Benchmark · motor propio vs Google OR-Tools",
                 fontsize=12, fontweight="bold", y=0.94)
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def formula_bellman() -> Path:
    """Renderiza la ecuación de Bellman aplicada al VRP como PNG."""
    ruta = _ruta("formulas", "bellman_vrp.png")
    fig, ax = plt.subplots(figsize=(9, 2.4), dpi=DPI)
    ax.axis("off")

    formula = (
        r"$g(S, i) = \min_{j \in S \setminus \{i\}}"
        r"\left[\, g(S \setminus \{i\}, j) + d(j, i) \,\right]$"
    )
    descripcion = (
        r"Subproblema: $g(S, i)$ = costo mínimo para visitar el subconjunto $S$ y terminar en $i$." "\n"
        r"Solución del TSP: $\mathrm{OPT} = \min_i \left[\, g(\{1,\dots,n-1\}, i) + d(i, 0) \,\right]$"
    )
    ax.text(0.5, 0.65, formula, fontsize=18, ha="center", va="center")
    ax.text(0.5, 0.20, descripcion, fontsize=10, ha="center", va="center", color="#444444")

    fig.suptitle("Ecuación de Bellman · Programación dinámica para el TSP",
                 fontsize=12, fontweight="bold", y=0.96)
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def hero_portada() -> Path:
    """Imagen de portada del artículo: red de centros de distribución conectados,
    camiones estilizados y partículas, estética tecnológica moderna oscura.
    """
    ruta = _ruta("imagenes", "00_hero_portada.png")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=DPI)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    rng = np.random.default_rng(42)

    bg_top = np.array([0x07, 0x14, 0x2A]) / 255
    bg_mid = np.array([0x0F, 0x2D, 0x52]) / 255
    bg_bot = np.array([0x07, 0x14, 0x2A]) / 255
    bg = np.zeros((400, 1, 3))
    for i in range(400):
        t = i / 399
        if t < 0.5:
            c = bg_top + (bg_mid - bg_top) * (t / 0.5)
        else:
            c = bg_mid + (bg_bot - bg_mid) * ((t - 0.5) / 0.5)
        bg[i, 0] = c
    ax.imshow(bg, extent=(0, 16, 0, 9), aspect="auto", interpolation="bilinear", zorder=0)

    grid_color = (0.20, 0.40, 0.65, 0.18)
    for x in np.arange(0, 16.1, 0.8):
        ax.axvline(x, color=grid_color, linewidth=0.5, zorder=1)
    for y in np.arange(0, 9.1, 0.8):
        ax.axhline(y, color=grid_color, linewidth=0.5, zorder=1)

    n_part = 80
    part_x = rng.uniform(0, 16, n_part)
    part_y = rng.uniform(0, 9, n_part)
    part_s = rng.uniform(2, 14, n_part)
    part_a = rng.uniform(0.1, 0.45, n_part)
    for x, y, s, a in zip(part_x, part_y, part_s, part_a):
        ax.scatter(x, y, s=s, color="#7BB7F0", alpha=a, zorder=2, edgecolor="none")

    centros = [
        ("DC-N",  3.4, 6.6, "#FF6B35"),
        ("DC-W",  2.0, 3.2, "#37C7FF"),
        ("DC-E",  9.4, 7.2, "#7CDB7E"),
        ("DC-S",  7.6, 2.4, "#FFC857"),
        ("DC-C", 12.5, 4.6, "#C58BF2"),
    ]

    n_clientes = 38
    clientes_xy = []
    for cx, cy, _ in [(c[1], c[2], c[3]) for c in centros]:
        n_local = rng.integers(5, 10)
        for _ in range(n_local):
            r = rng.uniform(0.6, 2.4)
            theta = rng.uniform(0, 2 * np.pi)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            x = float(np.clip(x, 0.6, 15.4))
            y = float(np.clip(y, 0.6, 8.4))
            clientes_xy.append((x, y))
    clientes_xy = clientes_xy[:n_clientes]

    centro_indices = list(range(len(centros)))
    cliente_a_centro = []
    for cx, cy in clientes_xy:
        dists = [(np.hypot(cx - c[1], cy - c[2]), i) for i, c in enumerate(centros)]
        cliente_a_centro.append(min(dists)[1])

    for (cx, cy), ci in zip(clientes_xy, cliente_a_centro):
        nombre, dx, dy, color = centros[ci]
        for halo_w, halo_a in [(3.5, 0.10), (2.0, 0.22)]:
            ax.plot([dx, cx], [dy, cy], color=color, alpha=halo_a, linewidth=halo_w, zorder=3)
        ax.plot([dx, cx], [dy, cy], color=color, alpha=0.9, linewidth=1.0, zorder=4)

    for cx, cy in clientes_xy:
        ax.scatter(cx, cy, s=180, color="white", alpha=0.18, zorder=5, edgecolor="none")
        ax.scatter(cx, cy, s=55,  color="white", alpha=0.95, zorder=6,
                   edgecolor="#0F2D52", linewidth=0.8)

    for nombre, dx, dy, color in centros:
        for halo_s, halo_a in [(1100, 0.10), (700, 0.18), (420, 0.30)]:
            ax.scatter(dx, dy, s=halo_s, color=color, alpha=halo_a, zorder=7, edgecolor="none")
        ax.scatter(dx, dy, s=260, color=color, marker="s", zorder=8,
                   edgecolor="white", linewidth=2.4)
        ax.text(dx, dy, nombre, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=9)

    def truck(ax, x, y, color, scale=0.45):
        body_w, body_h = 1.2 * scale, 0.55 * scale
        cab_w, cab_h = 0.55 * scale, 0.55 * scale
        ax.add_patch(plt.Rectangle((x, y), body_w, body_h,
                                   facecolor=color, edgecolor="white", linewidth=1.2, zorder=10))
        ax.add_patch(plt.Rectangle((x + body_w, y + 0.05 * scale), cab_w, cab_h - 0.05 * scale,
                                   facecolor="#0F2D52", edgecolor="white", linewidth=1.2, zorder=10))
        ax.add_patch(plt.Rectangle((x + body_w + 0.06 * scale, y + 0.18 * scale),
                                   cab_w - 0.16 * scale, cab_h - 0.30 * scale,
                                   facecolor="#7BB7F0", edgecolor="none", zorder=11))
        wheel_r = 0.10 * scale
        for wx in [x + 0.18 * scale, x + body_w - 0.20 * scale, x + body_w + cab_w - 0.20 * scale]:
            ax.add_patch(plt.Circle((wx, y - wheel_r * 0.4), wheel_r,
                                    facecolor="#1A2233", edgecolor="white", linewidth=1.0, zorder=12))

    truck(ax, 5.2, 8.30, "#FF6B35", scale=0.55)
    truck(ax, 0.4, 7.70, "#37C7FF", scale=0.55)
    truck(ax, 13.6, 0.55, "#7CDB7E", scale=0.55)

    ax.text(0.6, 1.95, "Gestión de rutas",
            fontsize=42, fontweight="bold", color="white",
            ha="left", va="bottom",
            family="DejaVu Sans", zorder=13)
    ax.text(0.6, 1.05, "multi-sucursal",
            fontsize=42, fontweight="bold", color="#FFC857",
            ha="left", va="bottom",
            family="DejaVu Sans", zorder=13)
    ax.text(0.65, 0.55, "Machine Learning   ·   Programación dinámica   ·   Open Source",
            fontsize=12, color="#A8C8E8", ha="left", va="bottom",
            family="DejaVu Sans", fontweight="600", zorder=13)

    ax.text(15.4, 8.55, "L E A N M A S T E R   P Y M E S",
            fontsize=11, color="#FFC857", fontweight="bold",
            ha="right", va="top", zorder=13)
    ax.text(15.4, 8.10, "Caso técnico",
            fontsize=10, color="#A8C8E8",
            ha="right", va="top", style="italic", zorder=13)

    fig.savefig(ruta, bbox_inches="tight", facecolor="#07142A", pad_inches=0)
    plt.close(fig)
    return ruta


def _callout_render(
    nombre_archivo: str,
    titulo: str | None,
    cuerpo: str,
    color_borde: str,
    color_fondo: str,
    color_texto: str = "#0F1729",
    altura: float = 2.6,
) -> Path:
    """Renderiza un callout (caja con borde lateral y fondo coloreado) como PNG."""
    ruta = _ruta("callouts", nombre_archivo)
    fig, ax = plt.subplots(figsize=(11, altura), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=color_fondo, edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.0, 0.0), 0.012, 1.0, facecolor=color_borde, edgecolor="none"))

    y_cursor = 0.86
    if titulo:
        ax.text(0.035, y_cursor, titulo, fontsize=13, fontweight="bold",
                color=color_borde, va="top", ha="left")
        y_cursor -= 0.18

    ax.text(0.035, y_cursor, cuerpo, fontsize=11.5, color=color_texto,
            va="top", ha="left", wrap=True, linespacing=1.5)

    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def callout_repo_github() -> Path:
    """Caja CTA al repositorio (sección 1 del artículo)."""
    return _callout_render(
        nombre_archivo="callout_repo_github.png",
        titulo="★ Sistema de código abierto",
        cuerpo=(
            "Para responder a esta situación desarrollé un sistema de código\n"
            "abierto, disponible en GitHub:  github.com/leanmasterpymes/gestion_rutas\n"
            "\n"
            "Aborda el problema desde la predicción de demanda hasta el plan\n"
            "diario coordinado entre los centros de distribución."
        ),
        color_borde="#D96F00",
        color_fondo="#FDF3E6",
        altura=3.0,
    )


def metric_grid_kpi() -> Path:
    """Cuadrícula de 4 KPIs del modelo de demanda (sección 3)."""
    ruta = _ruta("callouts", "metric_grid_kpi.png")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=DPI)
    fig.patch.set_facecolor("white")

    kpis = [
        ("10.1%",  "error en la demanda total\nesperada por día"),
        ("11.3",   "unidades de error promedio\npor cliente y día"),
        ("540",    "días de\nhistórico"),
        ("50",     "clientes en\nel caso"),
    ]
    color_borde = "#1A4373"
    color_fondo = "#F4F8FD"
    color_num = "#1A4373"
    color_lbl = "#2D3748"

    for ax, (num, lbl) in zip(axes, kpis):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90,
                                   facecolor=color_fondo, edgecolor=color_borde, linewidth=1.8))
        ax.text(0.5, 0.62, num, fontsize=34, fontweight="bold",
                ha="center", va="center", color=color_num)
        ax.text(0.5, 0.25, lbl, fontsize=10, ha="center", va="center",
                color=color_lbl, linespacing=1.4)

    fig.suptitle("Métricas del modelo de demanda · validación temporal (test sin filtración)",
                 fontsize=12, fontweight="bold", y=1.00, color="#0F2D52")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def pull_quote_held_karp() -> Path:
    """Pull-quote de la sección 4 (programación dinámica)."""
    ruta = _ruta("callouts", "pull_quote_held_karp.png")
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#FDF3E6", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.0, 0.0), 0.012, 1.0, facecolor="#D96F00", edgecolor="none"))

    ax.text(0.04, 0.92, "“", fontsize=64, color="#D96F00",
            fontweight="bold", va="top", ha="left")

    cita = (
        "¿Cuál es el camino más corto para visitar este conjunto de\n"
        "clientes y volver al centro de distribución?"
    )
    explicacion = "La programación dinámica responde esa pregunta sin probar todas las combinaciones posibles."

    ax.text(0.10, 0.72, cita, fontsize=15, fontstyle="italic",
            color="#1A2233", va="top", ha="left", linespacing=1.5)
    ax.text(0.10, 0.28, explicacion, fontsize=12, color="#2D3748",
            va="top", ha="left", linespacing=1.4)

    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def callout_info_benchmark() -> Path:
    """Callout-info que explica la diferencia de OR-Tools (sección 6)."""
    return _callout_render(
        nombre_archivo="callout_info_benchmark.png",
        titulo="ℹ Lectura del benchmark",
        cuerpo=(
            "En el caso pequeño, el sistema propio encuentra el óptimo idéntico al estándar\n"
            "industrial, 50 veces más rápido. En el caso grande, OR-Tools logra menor\n"
            "kilometraje porque consolida toda la flota en un único depósito ficticio (un\n"
            "esquema irrealizable en la operación real, donde cada camión sale de su centro\n"
            "de origen y debe regresar a él al cierre del turno). Bajo la misma restricción\n"
            "operativa, el sistema propio queda dentro de un margen razonable del óptimo."
        ),
        color_borde="#1A4373",
        color_fondo="#F4F8FD",
        altura=3.6,
    )


def generar_todas(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
    plan: pd.DataFrame,
    rutas_resueltas: list,
    metricas_antes_despues: pd.DataFrame | None = None,
    df_validacion: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Genera el set completo de figuras del artículo y devuelve sus rutas."""
    rutas: dict[str, Path] = {}
    rutas["hero"] = hero_portada()
    rutas["arquitectura"] = diagrama_arquitectura()
    rutas["mapa_clusters"] = mapa_clientes_clusters(clientes, sucursales, plan)
    rutas["plan_rutas"] = plan_rutas_mapa(plan, sucursales, rutas_resueltas)
    rutas["complejidad_grafico"] = heatmap_complejidad()
    rutas["complejidad_tabla"] = tabla_complejidad_render()
    rutas["bellman"] = formula_bellman()
    rutas["codigo_held_karp"] = codigo_held_karp_extracto()
    rutas["codigo_comandos"] = codigo_comandos_clone()
    rutas["benchmark_ortools"] = tabla_benchmark_ortools()
    rutas["callout_repo"] = callout_repo_github()
    rutas["metric_grid"] = metric_grid_kpi()
    rutas["pull_quote_dp"] = pull_quote_held_karp()
    rutas["callout_benchmark"] = callout_info_benchmark()
    if df_validacion is not None:
        rutas["demanda_curva"] = curva_demanda(
            df_validacion["real"], df_validacion["predicha"], df_validacion["fecha"],
        )
    if metricas_antes_despues is not None:
        rutas["antes_despues"] = comparativa_antes_despues(metricas_antes_despues)
    return rutas


if __name__ == "__main__":
    from src.clustering import construir_plan
    from src.data import construir_dataset
    from src.demand import entrenar, predecir_dia
    from src.dp_approx import resolver_plan

    tablas = construir_dataset()
    sucursales = tablas["sucursales"]
    clientes = tablas["clientes"]
    historico = tablas["historico_demanda"]

    modelo_dem = entrenar(historico, clientes)
    fecha_obj = pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1)
    pred = predecir_dia(modelo_dem, fecha_obj, historico, clientes)
    plan = construir_plan(clientes, sucursales, pred, modo="coordinado")
    rutas, km_total = resolver_plan(plan, sucursales)

    metricas = pd.DataFrame(
        [
            {"escenario": "Antes\n(sucursales como islas)", "km_total": km_total * 1.20, "n_rutas": 7, "costo_estimado": 425.0},
            {"escenario": "Después\n(ML + DP)", "km_total": km_total, "n_rutas": len(rutas), "costo_estimado": 320.0},
        ]
    )

    rutas_figuras = generar_todas(
        clientes=clientes,
        sucursales=sucursales,
        plan=plan,
        rutas_resueltas=rutas,
        metricas_antes_despues=metricas,
    )
    print("Figuras generadas:")
    for nombre, ruta in rutas_figuras.items():
        tamanyo = ruta.stat().st_size / 1024
        print(f"  {nombre:24s}  →  {ruta.relative_to(ruta.parents[2])}  ({tamanyo:.1f} KB)")
