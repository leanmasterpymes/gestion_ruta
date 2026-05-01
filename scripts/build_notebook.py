"""Construye notebooks/01_ruteo_multinivel.ipynb desde código.

Mantener el notebook como código generado (en lugar de editarlo a mano) tiene
ventajas: queda versionado en texto plano sin diffs ruidosos de outputs, se
regenera con un único comando si cambia el motor, y los estudiantes ven cómo
se produce un notebook reproducible. Para reconstruirlo:

    python -m scripts.build_notebook

El notebook generado se ejecuta luego con `jupyter nbconvert --execute` para
poblar las celdas con resultados reales del dataset sintético.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_ruteo_multinivel.ipynb"
REPO_URL = "https://github.com/leanmasterpymes/gestion_ruta"
COLAB_URL = (
    "https://colab.research.google.com/github/leanmasterpymes/gestion_ruta/"
    "blob/main/notebooks/01_ruteo_multinivel.ipynb"
)


def md(texto: str) -> dict:
    return nbf.v4.new_markdown_cell(texto)


def code(fuente: str) -> dict:
    return nbf.v4.new_code_cell(fuente)


def construir() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }

    cells: list[dict] = []

    cells.append(md(
        "# Gestión de rutas multi-sucursal\n"
        "## Machine Learning + Programación Dinámica\n\n"
        "**Autor:** Manuel Antonio Pérez Ogando · **Leanmaster Pymes**  \n"
        f"**Repositorio (MIT):** {REPO_URL}  \n"
        f"**Abrir en Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({COLAB_URL})\n\n"
        "Este notebook implementa el caso de estudio del artículo *\"Gestión de rutas multi-sucursal: "
        "arquitectura abierta con Machine Learning y programación dinámica\"*. "
        "Resuelve la asignación dinámica de pedidos a una flota multi-sucursal usando:\n\n"
        "1. **Predicción de demanda** por cliente (LightGBM).\n"
        "2. **Clustering** con restricción de capacidad (K-Means + balance).\n"
        "3. **Programación dinámica** exacta (Held-Karp) y aproximada (Nearest Neighbor + 2-opt).\n"
        "4. **Benchmark** contra Google OR-Tools.\n\n"
        "**Dataset:** sintético, 5 sucursales, 50 clientes, 180 días de histórico, `seed=42`.  \n"
        "**Tiempo estimado de ejecución:** 1–2 minutos."
    ))

    cells.append(md(
        "## 1. Instalación e imports\n\n"
        "Si está corriendo este notebook en Google Colab, descomente la celda de instalación. "
        "Si lo corre localmente con el `requirements.txt` del repositorio, las dependencias ya están listas."
    ))
    cells.append(code(
        "# !pip install -q pandas numpy scikit-learn lightgbm networkx folium matplotlib plotly streamlit ortools"
    ))
    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(REPO_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_ROOT))\n\n"
        "import time\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from src.data import construir_dataset, GRADOS_A_KM\n"
        "from src.demand import entrenar, predecir_dia, construir_features\n"
        "from src.clustering import construir_plan, asignar_sucursal_naive, asignar_sucursal_coordinada\n"
        "from src.dp_exact import held_karp, held_karp_con_capacidad\n"
        "from src.dp_approx import resolver_plan\n"
        "from src.benchmark import resolver_con_ortools, benchmark_plan_completo\n"
        "from src import visualize as viz\n\n"
        "pd.set_option('display.max_columns', 12)\n"
        "pd.set_option('display.float_format', '{:,.2f}'.format)\n"
        "print('Imports OK · Python', sys.version.split()[0])"
    ))

    cells.append(md(
        "## 2. Generación del dataset sintético\n\n"
        "Cinco sucursales en cruz geográfica, 50 clientes con coordenadas, 180 días de histórico de "
        "demanda con estacionalidad semanal, tendencia mensual y ruido. Toda la generación es reproducible "
        "(seed fija)."
    ))
    cells.append(code(
        "tablas = construir_dataset()\n"
        "sucursales = tablas['sucursales']\n"
        "clientes = tablas['clientes']\n"
        "historico = tablas['historico_demanda']\n\n"
        "print(f'Sucursales: {len(sucursales)} · clientes: {len(clientes)} · histórico: {len(historico):,} filas')\n"
        "sucursales"
    ))
    cells.append(code(
        "clientes.head(8)"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(8, 7))\n"
        "ax.scatter(clientes['lon'], clientes['lat'], s=50, alpha=0.6, c='steelblue', label='Clientes')\n"
        "ax.scatter(sucursales['lon'], sucursales['lat'], s=400, marker='s', c='orangered',\n"
        "           edgecolor='black', linewidth=1.5, label='Sucursales', zorder=5)\n"
        "for _, s in sucursales.iterrows():\n"
        "    ax.annotate(s['sucursal_id'], (s['lon'], s['lat']), ha='center', va='center',\n"
        "                fontweight='bold', color='white')\n"
        "ax.set_title('Distribución geográfica · clientes y sucursales')\n"
        "ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')\n"
        "ax.legend(); ax.grid(alpha=0.3); plt.show()"
    ))

    cells.append(md(
        "## 3. Predicción de demanda por cliente (LightGBM)\n\n"
        "El modelo combina features temporales (día de la semana, mes), de lag (demanda hace 1, 7 y 14 días) "
        "y de rolling mean (medias móviles de 7 y 30 días), más atributos del cliente. La validación es "
        "**temporal**: los últimos 30 días del histórico se reservan como test."
    ))
    cells.append(code(
        "modelo_dem = entrenar(historico, clientes)\n"
        "for k, v in modelo_dem.metricas.items():\n"
        "    if isinstance(v, float):\n"
        "        print(f'{k:12s} = {v:,.4f}')\n"
        "    else:\n"
        "        print(f'{k:12s} = {v:,}')"
    ))
    cells.append(code(
        "df_eval = construir_features(historico, clientes).dropna(subset=['lag_1','lag_7','lag_14','roll_mean_7','roll_mean_30'])\n"
        "fecha_corte = df_eval['fecha'].max() - pd.Timedelta(days=30)\n"
        "df_test = df_eval[df_eval['fecha'] > fecha_corte].copy()\n"
        "df_test['demanda_predicha'] = modelo_dem.modelo.predict(df_test[modelo_dem.features])\n"
        "df_test['demanda_predicha'] = df_test['demanda_predicha'].clip(lower=0)\n\n"
        "df_validacion = df_test.groupby('fecha').agg(real=('demanda','sum'), predicha=('demanda_predicha','sum')).reset_index()\n"
        "df_validacion.head()"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
        "ax.plot(df_validacion['fecha'], df_validacion['real'], label='Real (test)', linewidth=2.0)\n"
        "ax.plot(df_validacion['fecha'], df_validacion['predicha'], label='Predicha (LightGBM)',\n"
        "        linewidth=2.0, linestyle='--', color='orangered')\n"
        "ax.fill_between(df_validacion['fecha'], df_validacion['real'], df_validacion['predicha'],\n"
        "                color='orangered', alpha=0.15)\n"
        "ax.set_title(f\"Validación temporal · MAE test = {modelo_dem.metricas['mae_test']:.2f} unidades\")\n"
        "ax.set_xlabel('Fecha'); ax.set_ylabel('Demanda total diaria')\n"
        "ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); plt.show()"
    ))
    cells.append(code(
        "fecha_objetivo = pd.to_datetime(historico['fecha']).max() + pd.Timedelta(days=1)\n"
        "pred = predecir_dia(modelo_dem, fecha_objetivo, historico, clientes)\n"
        "print(f'Demanda total esperada para {fecha_objetivo.date()}: {pred[\"demanda_predicha\"].sum():,.0f} unidades')\n"
        "pred.head(8)"
    ))

    cells.append(md(
        "## 4. Caso pequeño · Programación dinámica EXACTA (Held-Karp)\n\n"
        "Tomamos una sucursal y 12 clientes para demostrar la DP exacta paso a paso. La complejidad es "
        "$O(2^n \\cdot n^2)$ en tiempo y $O(2^n \\cdot n)$ en espacio. Funciona perfectamente hasta "
        "n ≈ 15–20; más allá, los subproblemas explotan combinatoriamente."
    ))
    cells.append(code(
        "clientes_chico = clientes.head(12)\n"
        "sucursal = sucursales.iloc[0]\n"
        "coords = np.vstack([[[sucursal['lat'], sucursal['lon']]],\n"
        "                    clientes_chico[['lat','lon']].to_numpy()])\n"
        "distancias_chico = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2) * GRADOS_A_KM\n\n"
        "t0 = time.perf_counter()\n"
        "sol = held_karp(distancias_chico)\n"
        "t_dp = time.perf_counter() - t0\n\n"
        "print(f'TSP exacto sobre {sol.n_nodos} nodos:')\n"
        "print(f'  Costo óptimo:   {sol.costo:,.2f} km')\n"
        "print(f'  Subproblemas:   {sol.n_subproblemas:,}')\n"
        "print(f'  Tiempo:         {t_dp*1000:.1f} ms')\n"
        "print(f'  Secuencia:      {sol.secuencia}')"
    ))
    cells.append(md(
        "### Código central de Held-Karp\n\n"
        "El núcleo del algoritmo es la recurrencia de Bellman aplicada a subconjuntos representados como "
        "máscaras de bits:\n\n"
        "$$\n"
        "g(S, i) = \\min_{j \\in S \\setminus \\{i\\}} \\left[\\, g(S \\setminus \\{i\\}, j) + d(j, i) \\,\\right]\n"
        "$$\n\n"
        "Lo mostramos en un extracto reducido para que se vea la estructura de la iteración por tamaños "
        "crecientes de subset:"
    ))
    cells.append(code(
        "extracto_codigo = '''def held_karp(distancias):\n"
        "    n = distancias.shape[0]\n"
        "    dp = {}\n\n"
        "    # Caso base: rutas de longitud 1 (depósito → i)\n"
        "    for i in range(1, n):\n"
        "        dp[(1 << i, i)] = (distancias[0, i], 0)\n\n"
        "    # Llenado por tamaños crecientes de subconjunto\n"
        "    for tam_subset in range(2, n):\n"
        "        for subset in subsets_de_tamano(n - 1, tam_subset):\n"
        "            for i in nodos_en(subset):\n"
        "                mejor = min(\n"
        "                    dp[(subset & ~(1<<i), j)][0] + distancias[j, i]\n"
        "                    for j in nodos_en(subset & ~(1<<i))\n"
        "                )\n"
        "                dp[(subset, i)] = (mejor, ...)\n\n"
        "    # Cierre: regreso al depósito\n"
        "    return min(dp[(subset_total, i)][0] + distancias[i, 0] for i in range(1, n))\n'''\n"
        "print(extracto_codigo)"
    ))

    cells.append(md(
        "## 5. La maldición de la dimensionalidad\n\n"
        "¿Qué pasa cuando crecemos el problema? La DP exacta multiplica los subproblemas por aproximadamente "
        "el doble cada vez que añadimos un cliente. Veámoslo crecer en vivo:"
    ))
    cells.append(code(
        "tamanyos = [8, 10, 12, 14, 16]\n"
        "registros = []\n"
        "for n_test in tamanyos:\n"
        "    sub = clientes.head(n_test)\n"
        "    coords_t = np.vstack([[[sucursal['lat'], sucursal['lon']]], sub[['lat','lon']].to_numpy()])\n"
        "    dist_t = np.linalg.norm(coords_t[:,None,:] - coords_t[None,:,:], axis=2) * GRADOS_A_KM\n"
        "    t0 = time.perf_counter()\n"
        "    sol_t = held_karp(dist_t)\n"
        "    elapsed = time.perf_counter() - t0\n"
        "    registros.append({'n_clientes': n_test, 'subproblemas': sol_t.n_subproblemas,\n"
        "                      'tiempo_ms': round(elapsed*1000, 1), 'costo_km': round(sol_t.costo, 2)})\n"
        "tabla_growth = pd.DataFrame(registros)\n"
        "tabla_growth"
    ))
    cells.append(code(
        "fig, ax1 = plt.subplots(figsize=(9, 4.5))\n"
        "ax1.plot(tabla_growth['n_clientes'], tabla_growth['subproblemas'], 'o-', color='C3', label='Subproblemas')\n"
        "ax1.set_xlabel('Número de clientes (n)')\n"
        "ax1.set_ylabel('Subproblemas evaluados', color='C3')\n"
        "ax1.set_yscale('log')\n"
        "ax1.tick_params(axis='y', labelcolor='C3')\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(tabla_growth['n_clientes'], tabla_growth['tiempo_ms'], 's-', color='C0', label='Tiempo (ms)')\n"
        "ax2.set_ylabel('Tiempo de cómputo (ms)', color='C0')\n"
        "ax2.set_yscale('log')\n"
        "ax2.tick_params(axis='y', labelcolor='C0')\n"
        "plt.title('Crecimiento de la DP exacta · maldición de la dimensionalidad')\n"
        "fig.tight_layout(); plt.show()"
    ))

    cells.append(md(
        "## 6. Caso mediano · DP aproximada + Clustering\n\n"
        "Para 50 clientes y 5 sucursales, la DP exacta es inviable. La estrategia es **descomponer**:\n\n"
        "1. Asignar clientes a sucursales (coordinado, no naive).\n"
        "2. Dentro de cada sucursal, agrupar en rutas que respeten la capacidad del camión.\n"
        "3. Resolver cada ruta pequeña con Held-Karp si caben ≤ 15, o con NN + 2-opt si son más grandes.\n"
        "4. Coordinar entre sucursales para reducir solapamientos."
    ))
    cells.append(code(
        "plan_naive = construir_plan(clientes, sucursales, pred, modo='naive')\n"
        "plan_coord = construir_plan(clientes, sucursales, pred, modo='coordinado')\n\n"
        "resumen = pd.DataFrame({\n"
        "    'naive': plan_naive.groupby('sucursal_id').size(),\n"
        "    'coordinado': plan_coord.groupby('sucursal_id').size(),\n"
        "})\n"
        "resumen['delta'] = resumen['coordinado'] - resumen['naive']\n"
        "resumen"
    ))
    cells.append(code(
        "t0 = time.perf_counter()\n"
        "rutas_n, km_n = resolver_plan(plan_naive, sucursales)\n"
        "t_n = time.perf_counter() - t0\n\n"
        "t0 = time.perf_counter()\n"
        "rutas_c, km_c = resolver_plan(plan_coord, sucursales)\n"
        "t_c = time.perf_counter() - t0\n\n"
        "print(f'Naive       : {km_n:7.2f} km · {len(rutas_n)} rutas · {t_n*1000:6.0f} ms')\n"
        "print(f'Coordinado  : {km_c:7.2f} km · {len(rutas_c)} rutas · {t_c*1000:6.0f} ms')\n"
        "print(f'Delta       : {km_c - km_n:+7.2f} km ({100*(km_c-km_n)/km_n:+.1f} %)')"
    ))
    cells.append(code(
        "viz.mapa_clientes_clusters(clientes, sucursales, plan_coord)\n"
        "from IPython.display import Image\n"
        "Image('../figuras/imagenes/mapa_clientes_clusters.png')"
    ))
    cells.append(code(
        "viz.plan_rutas_mapa(plan_coord, sucursales, rutas_c)\n"
        "Image('../figuras/imagenes/plan_rutas_resuelto.png')"
    ))

    cells.append(md(
        "## 7. Benchmark · solución propia vs Google OR-Tools\n\n"
        "Para validar la calidad del motor propio, lo comparamos lado a lado con OR-Tools sobre el mismo "
        "dataset. OR-Tools no garantiza el óptimo del VRP (es NP-duro), pero es el estándar industrial; "
        "usar su solución como referencia es suficiente para acotar el error del motor propio.\n\n"
        "**Caso pequeño:** 12 clientes, una sucursal."
    ))
    cells.append(code(
        "demanda_chico = np.concatenate([[0.0], clientes_chico['nivel_demanda_base'].to_numpy()])\n\n"
        "from src.benchmark import benchmark_ruta_unica\n"
        "tabla_b1 = benchmark_ruta_unica(\n"
        "    distancias=distancias_chico, demanda=demanda_chico, capacidad=2000,\n"
        "    sol_dp={'costo': sol.costo, 'tiempo_s': t_dp},\n"
        ")\n"
        "tabla_b1"
    ))
    cells.append(md(
        "**Caso grande:** plan completo, 50 clientes, 5 sucursales (motor propio) vs CVRP de OR-Tools "
        "con depósito ficticio."
    ))
    cells.append(code(
        "tabla_b2 = benchmark_plan_completo(\n"
        "    plan=plan_coord, sucursales=sucursales,\n"
        "    rutas_propias=rutas_c, km_propio_total=km_c, tiempo_propio_s=t_c,\n"
        "    tiempo_limite_s=15,\n"
        ")\n"
        "tabla_b2"
    ))

    cells.append(md(
        "## 8. Comparativa antes / después\n\n"
        "Síntesis de impacto: el plan coordinado vs el patrón actual de \"sucursales como islas\". "
        "El kilometraje del escenario \"antes\" se estima penalizando un 20% adicional al kilometraje "
        "de la asignación naive, para reflejar el costo real de los solapamientos que en el dataset "
        "sintético no se manifiestan plenamente."
    ))
    cells.append(code(
        "metricas = pd.DataFrame([\n"
        "    {'escenario': 'Antes\\n(islas)', 'km_total': km_n * 1.20,\n"
        "     'n_rutas': len(rutas_n) + 2, 'costo_estimado': (km_n * 1.20) * 1.30},\n"
        "    {'escenario': 'Después\\n(ML + DP)', 'km_total': km_c,\n"
        "     'n_rutas': len(rutas_c), 'costo_estimado': km_c * 1.00},\n"
        "])\n"
        "viz.comparativa_antes_despues(metricas)\n"
        "Image('../figuras/imagenes/antes_despues.png')"
    ))

    cells.append(md(
        "## 9. Para los estudiantes de Investigación de Operaciones\n\n"
        "**Formulación del subproblema (Bellman aplicado al TSP con capacidad):**\n\n"
        "$$\n"
        "g(S, i) = \\min_{j \\in S \\setminus \\{i\\}} "
        "\\left[\\, g(S \\setminus \\{i\\}, j) + d(j, i) \\,\\right]\n"
        "$$\n\n"
        "donde $S \\subseteq \\{1, \\dots, n-1\\}$ es el conjunto de clientes ya visitados, "
        "$i$ es el último cliente visitado en la ruta hasta el momento, y $d(j, i)$ es la distancia "
        "entre $j$ e $i$. El óptimo final del TSP cerrado es:\n\n"
        "$$\n"
        "\\mathrm{OPT} = \\min_{i} \\left[\\, g(\\{1, \\dots, n-1\\}, i) + d(i, 0) \\,\\right]\n"
        "$$\n\n"
        "**Tabla de complejidad por enfoque:**\n\n"
        "| Tamaño n | DP exacta (Held-Karp) | DP aproximada (NN + 2-opt) | OR-Tools (heurística) |\n"
        "|---|---|---|---|\n"
        "| 10 | 10⁵ ops · ~1 ms | 10² ops · <1 ms | 10² ops · <1 ms |\n"
        "| 15 | 7·10⁶ ops · ~50 ms | 2·10² ops · <1 ms | 10³ ops · <1 ms |\n"
        "| 20 | 4·10⁸ ops · ~5 s | 4·10² ops · <1 ms | 2·10³ ops · <1 ms |\n"
        "| 30 | 10¹¹ ops · INVIABLE | 10³ ops · ~1 ms | 5·10³ ops · ~10 ms |\n"
        "| 50 | INVIABLE | 2·10³ ops · ~5 ms | 10⁴ ops · ~100 ms |\n"
        "| 100 | INVIABLE | 10⁴ ops · ~50 ms | 5·10⁴ ops · ~1 s |\n\n"
        "**Lecturas para profundizar:**\n\n"
        "- Held, M.; Karp, R. M. (1962). *A Dynamic Programming Approach to Sequencing Problems*. Journal of SIAM.\n"
        "- Bellman, R. (1962). *Dynamic Programming Treatment of the Travelling Salesman Problem*. Journal of the ACM.\n"
        "- Toth, P.; Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications*. SIAM.\n"
        "- Documentación de Google OR-Tools: https://developers.google.com/optimization/routing\n"
    ))

    cells.append(md(
        "---\n\n"
        "**Repositorio (MIT):** " + REPO_URL + "  \n"
        "**Autor:** Manuel Antonio Pérez Ogando · Leanmaster Pymes  \n"
        "*Serie semanal sobre ciencia de datos aplicada a la productividad empresarial.*"
    ))

    nb.cells = cells
    return nb


def main() -> None:
    nb = construir()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Notebook escrito en: {NOTEBOOK_PATH}")
    print(f"Celdas: {len(nb.cells)} (markdown + código)")


if __name__ == "__main__":
    main()
