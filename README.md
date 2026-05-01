# Gestión de rutas multi-sucursal — Machine Learning + Programación Dinámica

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/leanmasterpymes/gestion_ruta?style=social)](https://github.com/leanmasterpymes/gestion_ruta/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/leanmasterpymes/gestion_ruta?style=social)](https://github.com/leanmasterpymes/gestion_ruta/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/leanmasterpymes/gestion_ruta?style=social)](https://github.com/leanmasterpymes/gestion_ruta/watchers)
[![Last commit](https://img.shields.io/github/last-commit/leanmasterpymes/gestion_ruta)](https://github.com/leanmasterpymes/gestion_ruta/commits)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leanmasterpymes/gestion_ruta/blob/main/notebooks/01_ruteo_multinivel.ipynb)

> Arquitectura abierta y reproducible para que cualquier distribuidora con varias sucursales deje de planificar rutas como islas y empiece a operar un plan diario coordinado, alimentado por **predicción de demanda** y **clustering de clientes** con Machine Learning, y resuelto con **programación dinámica** sobre código Python puro.

---

## Demo en vivo · sin instalación

- **[Planificador interactivo en Streamlit](https://gestionruta-fvzdvkcjthzfavm9fcmk9c.streamlit.app/):** plan diario coordinado entre centros de distribución, con mapa interactivo, métricas del modelo de demanda, comparativa antes / después y descarga del plan en CSV.
- **⭐ Repositorio (código abierto · MIT):** clónelo y ejecute el stack en su computadora.
- **Notebook ejecutable en Colab:** algoritmo paso a paso, usa el badge "Open in Colab" de arriba.

---

## ¿Qué resuelve este proyecto?

En la mayoría de las distribuidoras con varias sucursales, dos camiones de la misma empresa terminan a tres cuadras del mismo cliente el mismo día. Cada sucursal planifica su ruta como una isla, sin saber qué hace la sucursal vecina. El resultado: kilómetros duplicados, combustible perdido, horas extra del chofer y subcontrataciones flash de camiones cuando la demanda real supera la capacidad asignada.

Este sistema combina:

1. **Predicción de demanda por cliente** (regresión con LightGBM/XGBoost) que estima cuánto pedirá cada cliente en el horizonte de planificación, con un nivel de error medible.
2. **Clustering inteligente de clientes** (K-Means con restricción de capacidad) que agrupa la demanda en zonas balanceadas por camión.
3. **Programación dinámica** (Held-Karp para el caso pequeño y DP aproximada por cluster para el caso mediano) que resuelve la secuencia óptima de cada ruta minimizando kilómetros y respetando ventanas horarias.
4. **Benchmark contra Google OR-Tools** para comparar la calidad de la solución propia con el estándar de la industria, sobre el mismo dataset.

El resultado es un plan diario coordinado entre sucursales, con cifras reproducibles a partir de un dataset sintético con seed fija.

---

## Estructura del repositorio

```
gestion_ruta/
├── app/                     Aplicación Streamlit (planificador interactivo)
├── data/                    Datasets sintéticos (CSV/Parquet) generados con seed fija
├── docs/                    Artículo en sus tres formatos:
│                              articulo_linkedin.html  (versión LinkedIn con visuales)
│                              articulo_linkedin.docx  (texto plano + avisos para LinkedIn Articles)
│                              articulo_web.html       (versión extendida para web Leanmaster Pymes)
├── figuras/                 Imágenes referenciadas por el artículo
│   ├── codigo/              Capturas de bloques de código
│   ├── diagramas/           Arquitecturas, pipelines, flujos (SVG/PNG)
│   ├── formulas/            Ecuaciones renderizadas
│   ├── imagenes/            Mapas, capturas de demo, fotos
│   └── tablas/              Tablas renderizadas como imagen
├── notebooks/               Notebooks ejecutables en Colab
├── src/                     Módulos Python del motor
│   ├── data.py              Generación del dataset sintético
│   ├── demand.py            Predicción de demanda por cliente (ML)
│   ├── clustering.py        Clustering con restricción de capacidad
│   ├── dp_exact.py          Held-Karp (DP exacta) — caso pequeño
│   ├── dp_approx.py         DP aproximada por cluster — caso mediano
│   ├── benchmark.py         Comparación contra Google OR-Tools
│   └── visualize.py         Generación automática de figuras
├── LICENSE                  MIT
├── README.md                este archivo
└── requirements.txt         dependencias Python
```

---

## Cómo probar el sistema

### Opción 1 — Ejecución local

Requiere **Python 3.12** (también funciona en 3.11). Versiones más recientes (3.13, 3.14) pueden romper compatibilidad con `lightgbm` u `ortools`; el archivo `runtime.txt` fija la versión para Streamlit Cloud.

```bash
git clone https://github.com/leanmasterpymes/gestion_ruta
cd gestion_ruta
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La aplicación abre en `http://localhost:8501` con un dataset sintético cargado por defecto. El usuario puede subir su propio CSV de clientes para generar un plan personalizado.

### Opción 2 — Notebook en Colab

Haga clic en el badge **Open in Colab** del encabezado. Se abre el notebook `notebooks/01_ruteo_multinivel.ipynb` en una sesión gratuita de Google Colab, sin instalación local. Ejecute las celdas en orden para ver el algoritmo completo:

1. Generación del dataset sintético (5 sucursales, 50 clientes, 8 camiones).
2. Predicción de demanda con LightGBM.
3. Clustering de clientes con restricción de capacidad.
4. Resolución por DP exacta (caso pequeño) y DP aproximada (caso mediano).
5. Benchmark contra Google OR-Tools.
6. Visualización del plan resuelto sobre el mapa.

---

## Stack técnico

- **Lenguaje:** Python 3.10+
- **Datos:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (clustering), `lightgbm` (predicción demanda)
- **Optimización:** implementación propia de Held-Karp y DP aproximada, `networkx` (grafos), `ortools` (benchmark)
- **Visualización:** `matplotlib`, `plotly`, `folium`
- **Aplicación:** `streamlit`
- **Notebooks:** Jupyter / Google Colab

---

## Autor

**Manuel Antonio Pérez Ogando** — Ingeniero industrial, MSc en Gestión Estratégica para el Desarrollo de Software · Profesor de Investigación de Operaciones · Especialista en mapeo, análisis y mejora de procesos · Certificado **Lean Six Sigma Green Belt (LSSGB)** · Certificado en **Power BI** · Estudiante de **Matemática Aplicada, mención Informática** en la UASD.

*Leanmaster Pymes — entrega de la serie semanal sobre ciencia de datos aplicada a la productividad empresarial.*

---

## Licencia

Distribuido bajo licencia **MIT**. Vea el archivo [LICENSE](LICENSE) para los términos completos.

Cualquier PYME puede clonar, adaptar y operar este sistema en su empresa sin trabas legales, e incluso usarlo con fines comerciales internos.
