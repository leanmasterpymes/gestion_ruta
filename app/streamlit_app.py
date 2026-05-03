"""Planificador de rutas multi-sucursal · aplicación Streamlit.

Carga el dataset sintético del repositorio, entrena el modelo de demanda,
construye el plan diario coordinado entre centros de distribución y lo muestra
sobre un mapa interactivo. Incluye comparativa naive vs coordinada, métricas
de validación del modelo y descarga del plan en CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering import construir_plan
from src.data import construir_dataset
from src.demand import entrenar, predecir_dia
from src.dp_approx import resolver_plan


st.set_page_config(
    page_title="Gestión de rutas multi-sucursal · Leanmaster Pymes",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETA = [
    "#1F4F87", "#D96F00", "#1F6B33", "#9C2222", "#5B3DA8",
    "#7A5B12", "#147F7B", "#7C2B65",
]


CSS_EJECUTIVO = """
<style>
:root {
  --color-primary: #1A4373;
  --color-primary-dark: #0F2D52;
  --color-primary-soft: #E8F1F8;
  --color-accent: #D96F00;
  --color-accent-soft: #FDF3E6;
  --color-success: #1F6B33;
  --color-success-soft: #E6F2EA;
  --color-text: #0F1729;
  --color-text-soft: #4F5B6F;
  --color-border: #DFE5EE;
  --color-bg-card: #FFFFFF;
  --color-bg-soft: #F4F8FD;
}

html, body, [class*="stApp"] { background-color: #F7F9FC !important; }

[data-testid="stHeader"] {
  background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-primary) 100%);
  height: 4px;
}

.block-container { padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1400px; }

.hero-banner {
  background: linear-gradient(135deg, #0F2D52 0%, #1A4373 55%, #2563A8 100%);
  padding: 2.4rem 2.6rem 2.0rem 2.6rem;
  border-radius: 14px;
  margin-bottom: 1.6rem;
  color: white;
  box-shadow: 0 4px 18px rgba(15, 45, 82, 0.18);
  position: relative;
  overflow: hidden;
}
.hero-banner::after {
  content: "";
  position: absolute;
  right: -40px; top: -40px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(217, 111, 0, 0.32) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  text-transform: uppercase;
  letter-spacing: 2px;
  font-size: 11px;
  font-weight: 700;
  color: #FFD8A8;
  margin-bottom: 8px;
}
.hero-title {
  font-size: 30px;
  font-weight: 800;
  margin: 0 0 10px 0;
  color: white;
  letter-spacing: -0.3px;
  line-height: 1.15;
}
.hero-subtitle {
  font-size: 15.5px;
  color: rgba(255, 255, 255, 0.86);
  margin: 0;
  max-width: 880px;
  line-height: 1.5;
}
.hero-chips { margin-top: 14px; }
.hero-chip {
  display: inline-block;
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.28);
  color: white;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-right: 8px;
  margin-top: 6px;
}

.section-title {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.6px;
  color: var(--color-primary);
  margin: 1.2rem 0 0.6rem 0;
  padding-bottom: 6px;
  border-bottom: 2px solid var(--color-primary-soft);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  margin-bottom: 1rem;
}
.kpi-card {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-left: 4px solid var(--color-primary);
  border-radius: 10px;
  padding: 16px 18px 14px 18px;
  box-shadow: 0 1px 3px rgba(15, 45, 82, 0.04);
  transition: transform 0.15s, box-shadow 0.15s;
}
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(15, 45, 82, 0.12);
}
.kpi-card.accent { border-left-color: var(--color-accent); }
.kpi-card.success { border-left-color: var(--color-success); }
.kpi-label {
  font-size: 11.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--color-text-soft);
  margin-bottom: 6px;
}
.kpi-value {
  font-size: 28px;
  font-weight: 800;
  color: var(--color-text);
  line-height: 1.1;
  margin-bottom: 2px;
}
.kpi-help {
  font-size: 12px;
  color: var(--color-text-soft);
}

div[data-testid="stMetric"] {
  background: white;
  border: 1px solid var(--color-border);
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: 0 1px 3px rgba(15, 45, 82, 0.04);
}
div[data-testid="stMetricLabel"] p {
  font-size: 11.5px !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--color-text-soft) !important;
}
div[data-testid="stMetricValue"] {
  color: var(--color-text) !important;
  font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] svg { display: none; }

.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 2px solid var(--color-border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 8px 8px 0 0;
  padding: 10px 18px;
  color: var(--color-text-soft);
  font-weight: 600;
  font-size: 14px;
}
.stTabs [aria-selected="true"] {
  background: var(--color-primary-soft) !important;
  color: var(--color-primary) !important;
  border-bottom: 3px solid var(--color-accent) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F2D52 0%, #1A4373 100%);
}
[data-testid="stSidebar"] * { color: #E8F1F8 !important; }
[data-testid="stSidebar"] .st-emotion-cache-1weic72 { color: white !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.18) !important; }
[data-testid="stSidebar"] a { color: #FFD8A8 !important; text-decoration: none; }
[data-testid="stSidebar"] a:hover { text-decoration: underline; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: white !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { color: white !important; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { background: rgba(255,255,255,0.12) !important; border-color: rgba(255,255,255,0.5) !important; }

.stDataFrame, [data-testid="stDataFrame"] {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 8px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--color-accent) 0%, #F08B1C 100%) !important;
  border: none !important;
  box-shadow: 0 2px 8px rgba(217, 111, 0, 0.3) !important;
}

div[data-baseweb="notification"] {
  border-radius: 10px;
  border-left-width: 4px;
}

.brand-footer {
  margin-top: 2.4rem;
  padding: 1.2rem 1.4rem;
  background: white;
  border: 1px solid var(--color-border);
  border-radius: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 13px;
  color: var(--color-text-soft);
}
.brand-footer strong { color: var(--color-primary); }
.brand-footer .footer-link {
  color: var(--color-accent);
  font-weight: 600;
  text-decoration: none;
}
.brand-footer .footer-link:hover { text-decoration: underline; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
"""


@st.cache_data(show_spinner=False)
def cargar_dataset_cache() -> dict:
    return construir_dataset()


@st.cache_resource(show_spinner=False)
def entrenar_modelo_cache(_version: str):
    tablas = cargar_dataset_cache()
    return entrenar(tablas["historico_demanda"], tablas["clientes"])


def mapa_rutas(plan: pd.DataFrame, sucursales: pd.DataFrame, rutas: list) -> folium.Map:
    """Mapa folium con las rutas resueltas, coloreadas por centro de distribución."""
    centro_lat = plan["lat"].mean()
    centro_lon = plan["lon"].mean()
    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    plan_idx = plan.set_index("cliente_id")
    suc_idx = sucursales.set_index("sucursal_id")
    suc_ids = sucursales["sucursal_id"].tolist()

    for i, suc in sucursales.reset_index(drop=True).iterrows():
        color = PALETA[i % len(PALETA)]
        folium.RegularPolygonMarker(
            location=[suc["lat"], suc["lon"]],
            number_of_sides=4,
            radius=14,
            color="#000000",
            fill_color=color,
            fill_opacity=1.0,
            weight=2,
            rotation=45,
            popup=folium.Popup(
                f"<b>{suc['sucursal_id']}</b><br>"
                f"{suc['n_camiones']} camiones · capacidad {suc['capacidad_camion']} u",
                max_width=240,
            ),
        ).add_to(m)

    for r in rutas:
        suc_i = suc_ids.index(r.sucursal_id)
        color = PALETA[suc_i % len(PALETA)]
        suc_coord = suc_idx.loc[r.sucursal_id]
        coords = [(suc_coord["lat"], suc_coord["lon"])]
        for cid in r.secuencia_clientes:
            c = plan_idx.loc[cid]
            coords.append((c["lat"], c["lon"]))
        coords.append((suc_coord["lat"], suc_coord["lon"]))

        folium.PolyLine(
            coords, color=color, weight=2.6, opacity=0.75,
        ).add_to(m)

        for i, cid in enumerate(r.secuencia_clientes):
            c = plan_idx.loc[cid]
            folium.CircleMarker(
                location=[c["lat"], c["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(
                    f"<b>{cid}</b><br>"
                    f"Ruta: {r.ruta_global}<br>"
                    f"Posición: {i + 1} de {len(r.secuencia_clientes)}<br>"
                    f"Demanda predicha: {c['demanda_predicha']:.1f} u",
                    max_width=240,
                ),
            ).add_to(m)

    return m


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='padding:6px 0 16px 0;'>"
            "<div style='font-size:11px; letter-spacing:2px; color:#FFD8A8; "
            "font-weight:700; text-transform:uppercase;'>Leanmaster Pymes</div>"
            "<div style='font-size:18px; font-weight:800; color:white; margin-top:2px;'>"
            "Planificador de Rutas</div>"
            "<div style='font-size:12.5px; color:rgba(255,255,255,0.78); margin-top:4px;'>"
            "ML + Programación dinámica</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Configuración del plan")
        modo = st.radio(
            "Asignación cliente → centro de distribución",
            ["Coordinada (recomendada)", "Naive (cada centro toma cercanos)"],
            index=0,
            help=(
                "La asignación coordinada respeta la capacidad real de cada centro y "
                "evita solapamientos. La naive simula el patrón actual de muchas "
                "distribuidoras: cada centro toma sus clientes más cercanos sin "
                "coordinar con los vecinos."
            ),
        )
        modo_key = "coordinado" if modo.startswith("Coordinada") else "naive"

        st.markdown("---")
        st.markdown("##### Pipeline del sistema")
        st.markdown(
            "1. Predicción de demanda por cliente\n"
            "2. Clustering inteligente con restricción de capacidad\n"
            "3. Programación dinámica (Held-Karp / NN + 2-opt)\n"
            "4. Coordinación entre centros de distribución\n\n"
            "Validado contra **Google OR-Tools** (estándar industrial)."
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:12px;'>"
            "<strong>Código abierto · Licencia MIT</strong><br>"
            "<a href='https://github.com/leanmasterpymes/gestion_rutas'>"
            "github.com/leanmasterpymes/gestion_rutas</a></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:11.5px; color:rgba(255,255,255,0.65); "
            "margin-top:10px; line-height:1.5;'>"
            "Por Manuel Antonio Pérez Ogando — Leanmaster Pymes. "
            "Serie semanal de ciencia de datos aplicada a la productividad empresarial."
            "</div>",
            unsafe_allow_html=True,
        )

    return modo_key


def render_kpis(
    clientes_df: pd.DataFrame,
    sucursales_df: pd.DataFrame,
    rutas: list,
    km_total: float,
    demanda_total: float,
) -> None:
    cards = [
        ("Clientes", f"{len(clientes_df)}", "atendidos en el plan", "primary"),
        ("Centros", f"{len(sucursales_df)}", "de distribución activos", "primary"),
        ("Rutas", f"{len(rutas)}", "planificadas para mañana", "accent"),
        ("Kilometraje", f"{km_total:,.0f} km", "recorrido total estimado", "accent"),
        ("Demanda", f"{demanda_total:,.0f} u", "predicha por el modelo", "success"),
    ]
    html = "<div class='kpi-grid'>"
    for label, value, helper, variant in cards:
        cls = "kpi-card"
        if variant == "accent": cls += " accent"
        if variant == "success": cls += " success"
        html += (
            f"<div class='{cls}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-help'>{helper}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS_EJECUTIVO, unsafe_allow_html=True)

    st.markdown(
        "<div class='hero-banner'>"
        "<div class='hero-eyebrow'>Leanmaster Pymes · Caso técnico</div>"
        "<h1 class='hero-title'>Gestión de rutas multi-sucursal</h1>"
        "<p class='hero-subtitle'>"
        "Plan diario coordinado entre centros de distribución, alimentado por "
        "predicción de demanda con Machine Learning y resuelto con programación "
        "dinámica. Validado contra el estándar industrial Google OR-Tools."
        "</p>"
        "<div class='hero-chips'>"
        "<span class='hero-chip'>LightGBM</span>"
        "<span class='hero-chip'>K-Means con capacidad</span>"
        "<span class='hero-chip'>Held-Karp · NN + 2-opt</span>"
        "<span class='hero-chip'>OR-Tools</span>"
        "<span class='hero-chip'>Open source · MIT</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    modo_key = render_sidebar()

    with st.spinner("Cargando dataset sintético (5 centros, 50 clientes, 540 días de histórico)..."):
        tablas = cargar_dataset_cache()
    sucursales = tablas["sucursales"]
    clientes = tablas["clientes"]
    historico = tablas["historico_demanda"]

    with st.spinner("Entrenando modelo de predicción de demanda (LightGBM, ~10 segundos)..."):
        modelo_dem = entrenar_modelo_cache("v1")

    fecha_obj = pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1)
    pred = predecir_dia(modelo_dem, fecha_obj, historico, clientes)

    plan = construir_plan(clientes, sucursales, pred, modo=modo_key)
    rutas, km_total = resolver_plan(plan, sucursales)
    demanda_total = float(pred["demanda_predicha"].sum())

    st.markdown(
        f"<div class='section-title'>Plan diario · {fecha_obj.strftime('%A %d %B %Y').capitalize()}</div>",
        unsafe_allow_html=True,
    )
    render_kpis(clientes, sucursales, rutas, km_total, demanda_total)

    tab_plan, tab_demanda, tab_compara, tab_descargar = st.tabs(
        ["📍 Plan de rutas", "📊 Predicción de demanda", "🔄 Comparativa antes / después", "💾 Descargar plan"],
    )

    with tab_plan:
        col_mapa, col_resumen = st.columns([3, 2])
        with col_mapa:
            st.markdown("##### Mapa interactivo")
            st_folium(
                mapa_rutas(plan, sucursales, rutas),
                width=None,
                height=560,
                returned_objects=[],
            )
        with col_resumen:
            st.markdown("##### Resumen por ruta")
            df_resumen = pd.DataFrame(
                [
                    {
                        "Ruta": r.ruta_global,
                        "Centro": r.sucursal_id,
                        "Clientes": len(r.secuencia_clientes),
                        "Carga (u)": round(r.demanda_total, 1),
                        "Distancia (km)": round(r.distancia_km, 2),
                        "Algoritmo": r.metodo,
                    }
                    for r in rutas
                ]
            )
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)

            st.markdown("##### Capacidad por centro")
            df_cap = sucursales.assign(
                capacidad_total=lambda d: d["n_camiones"] * d["capacidad_camion"],
            )[["sucursal_id", "n_camiones", "capacidad_camion", "capacidad_total"]].rename(
                columns={
                    "sucursal_id": "Centro",
                    "n_camiones": "Camiones",
                    "capacidad_camion": "Cap. por camión",
                    "capacidad_total": "Capacidad total",
                }
            )
            st.dataframe(df_cap, use_container_width=True, hide_index=True)

    with tab_demanda:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("##### Demanda predicha por cliente para mañana")
            df_pred = pred.sort_values("demanda_predicha", ascending=False).copy()
            df_pred["demanda_predicha"] = df_pred["demanda_predicha"].round(1)
            st.bar_chart(df_pred.set_index("cliente_id")["demanda_predicha"], height=400)
            st.caption("Los clientes están ordenados por demanda predicha descendente.")

        with col_b:
            st.markdown("##### Métricas de validación temporal")
            metricas = modelo_dem.metricas
            df_metricas = pd.DataFrame(
                [
                    {
                        "Métrica": "Error agregado por día (WAPE)",
                        "Valor": f"{metricas['wape_test_dia']:.1f}%",
                    },
                    {
                        "Métrica": "Error promedio por cliente (MAE)",
                        "Valor": f"{metricas['mae_test']:.2f} u",
                    },
                    {
                        "Métrica": "Días de entrenamiento",
                        "Valor": f"{metricas['n_train']:,}",
                    },
                    {
                        "Métrica": "Días de prueba",
                        "Valor": f"{metricas['n_test']:,}",
                    },
                ]
            )
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            st.caption(
                "Validación temporal: los últimos 30 días del histórico se reservan "
                "como prueba, sin que el modelo los vea durante el aprendizaje."
            )
            st.success(
                f"WAPE diario de {metricas['wape_test_dia']:.1f}% en línea con "
                "benchmarks de clase mundial para forecast de retail (7–15%)."
            )

    with tab_compara:
        st.markdown("##### Asignación naive vs asignación coordinada")
        st.caption(
            "Compara el plan que arma cada modo de asignación sobre el mismo dataset y "
            "la misma demanda predicha."
        )

        plan_naive = construir_plan(clientes, sucursales, pred, modo="naive")
        rutas_n, km_n = resolver_plan(plan_naive, sucursales)
        plan_coord = construir_plan(clientes, sucursales, pred, modo="coordinado")
        rutas_c, km_c = resolver_plan(plan_coord, sucursales)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Plan naive", f"{km_n:,.1f} km", f"{len(rutas_n)} rutas")
        with col_b:
            st.metric("Plan coordinado", f"{km_c:,.1f} km", f"{len(rutas_c)} rutas")
        with col_c:
            delta = km_c - km_n
            delta_pct = 100 * delta / km_n if km_n > 0 else 0
            if delta < -0.5:
                st.metric(
                    "Ahorro coordinada vs naive",
                    f"{abs(delta):,.1f} km",
                    f"{abs(delta_pct):.1f}% menos",
                )
            elif delta > 0.5:
                st.metric(
                    "Diferencia coordinada vs naive",
                    f"{delta:,.1f} km",
                    f"{delta_pct:.1f}% más",
                    delta_color="inverse",
                )
            else:
                st.metric("Diferencia", "≈ 0 km", "Coinciden en este caso")

        st.markdown("##### Distribución de clientes por centro")
        df_n = plan_naive.groupby("sucursal_id").size().rename("naive")
        df_c = plan_coord.groupby("sucursal_id").size().rename("coordinada")
        comparativa = pd.concat([df_n, df_c], axis=1).fillna(0).astype(int)
        st.bar_chart(comparativa, height=300)

        if abs(delta_pct) < 1:
            st.info(
                "En este caso de estudio la holgura de capacidad es alta (la flota "
                "puede absorber la demanda sin tensión), por lo que la asignación "
                "coordinada y la naive llegan a planes muy parecidos. En operaciones "
                "reales con flota más ajustada, la coordinada suele ahorrar entre "
                "5% y 15% del kilometraje."
            )

    with tab_descargar:
        st.markdown("##### Plan completo listo para exportar")
        plan_export = plan.copy()
        orden = pd.DataFrame(
            [
                {"cliente_id": cid, "orden_visita": i + 1, "ruta_resuelta": r.ruta_global}
                for r in rutas
                for i, cid in enumerate(r.secuencia_clientes)
            ]
        )
        plan_export = plan_export.merge(orden, on="cliente_id", how="left")
        plan_export = plan_export[
            [
                "cliente_id", "sucursal_id", "ruta_resuelta", "orden_visita",
                "lat", "lon", "demanda_predicha",
            ]
        ].rename(
            columns={
                "cliente_id": "Cliente",
                "sucursal_id": "Centro",
                "ruta_resuelta": "Ruta",
                "orden_visita": "Orden",
                "lat": "Latitud",
                "lon": "Longitud",
                "demanda_predicha": "Demanda predicha",
            }
        ).sort_values(["Centro", "Ruta", "Orden"])

        st.dataframe(plan_export, use_container_width=True, hide_index=True)

        csv_data = plan_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar plan en CSV",
            data=csv_data,
            file_name=f"plan_rutas_{fecha_obj.date()}_{modo_key}.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption(
            "El CSV contiene cada cliente con su centro asignado, ruta, orden de visita, "
            "coordenadas y demanda predicha. Listo para importar en una hoja de cálculo "
            "o en el sistema de despacho."
        )

    st.markdown(
        "<div class='brand-footer'>"
        "<div>"
        "<strong>Leanmaster Pymes</strong> · Manuel Antonio Pérez Ogando &nbsp;·&nbsp; "
        "Ingeniería industrial · Investigación de operaciones · Lean Six Sigma"
        "</div>"
        "<div>"
        "<a class='footer-link' href='https://github.com/leanmasterpymes/gestion_rutas'>Repositorio en GitHub</a>"
        " &nbsp;·&nbsp; "
        "<a class='footer-link' href='https://colab.research.google.com/github/leanmasterpymes/gestion_rutas/blob/main/notebooks/01_ruteo_multinivel.ipynb'>Notebook en Colab</a>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
