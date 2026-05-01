"""Generación del dataset sintético reproducible.

Produce 5 sucursales, 50 clientes y un histórico de 180 días de demanda diaria
con seed fija para que cualquier ejecución del notebook reproduzca exactamente
las mismas cifras del artículo. El dataset es genérico (agnóstico de industria)
y representa una distribuidora con cobertura urbana y semiurbana.

Convenciones:
    - Coordenadas en grados decimales (latitud, longitud) sobre un área ficticia
      centrada en (0.0, 0.0). Las distancias euclídeas en grados se convierten
      a kilómetros aproximados con un factor uniforme de 111 km/grado.
    - Capacidad del camión expresada en unidades genéricas de carga.
    - Demanda en unidades por cliente y día.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_SUCURSALES = 5
N_CLIENTES = 50
N_DIAS_HISTORICO = 540
GRADOS_A_KM = 111.0
RUIDO_DEMANDA = 0.06

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Config:
    """Configuración del generador de datasets sintéticos."""

    n_sucursales: int = N_SUCURSALES
    n_clientes: int = N_CLIENTES
    n_dias_historico: int = N_DIAS_HISTORICO
    seed: int = SEED


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generar_sucursales(config: Config = Config()) -> pd.DataFrame:
    """Devuelve las sucursales con coordenadas, capacidad de flota y horario.

    Las sucursales se distribuyen en una cruz geográfica: una central y las
    demás en los puntos cardinales, separadas ~22 km entre sí (suficiente para
    que sus áreas de cobertura se solapen y aparezca el problema de
    coordinación entre rutas).
    """
    nombres = [f"S{i+1}" for i in range(config.n_sucursales)]
    centros = np.array(
        [
            [0.00, 0.00],   # Central
            [0.20, 0.00],   # Norte
            [-0.20, 0.00],  # Sur
            [0.00, 0.20],   # Este
            [0.00, -0.20],  # Oeste
        ][: config.n_sucursales]
    )
    rng = _rng(config.seed)
    n_camiones = rng.integers(low=2, high=4, size=config.n_sucursales, endpoint=True)
    capacidad_camion = rng.choice([800, 1000, 1200], size=config.n_sucursales)
    apertura = np.full(config.n_sucursales, 7)
    cierre = np.full(config.n_sucursales, 18)

    return pd.DataFrame(
        {
            "sucursal_id": nombres,
            "lat": centros[:, 0],
            "lon": centros[:, 1],
            "n_camiones": n_camiones,
            "capacidad_camion": capacidad_camion,
            "hora_apertura": apertura,
            "hora_cierre": cierre,
        }
    )


def generar_clientes(config: Config = Config()) -> pd.DataFrame:
    """Devuelve los clientes con coordenadas, ventana horaria y tipo.

    Los clientes se distribuyen sobre un área de ~50x50 km centrada en el
    origen, con dos modos para que el clustering tenga estructura: 70% en
    cluster gaussiano alrededor del centro y 30% periferia uniforme.
    """
    rng = _rng(config.seed + 1)
    n_centro = int(round(config.n_clientes * 0.7))
    n_periferia = config.n_clientes - n_centro

    coords_centro = rng.normal(loc=0.0, scale=0.10, size=(n_centro, 2))
    coords_periferia = rng.uniform(low=-0.25, high=0.25, size=(n_periferia, 2))
    coords = np.vstack([coords_centro, coords_periferia])
    rng.shuffle(coords)

    tipos = rng.choice(
        ["minimarket", "restaurante", "ferreteria", "farmacia", "almacen"],
        size=config.n_clientes,
        p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )
    ventana_inicio = rng.choice([7, 8, 9, 10], size=config.n_clientes)
    ventana_fin = ventana_inicio + rng.choice([4, 5, 6, 7], size=config.n_clientes)
    ventana_fin = np.clip(ventana_fin, a_min=None, a_max=18)

    tiempo_servicio = rng.integers(low=5, high=20, size=config.n_clientes, endpoint=True)
    nivel_demanda = rng.gamma(shape=2.0, scale=15.0, size=config.n_clientes)

    return pd.DataFrame(
        {
            "cliente_id": [f"C{i+1:03d}" for i in range(config.n_clientes)],
            "lat": coords[:, 0],
            "lon": coords[:, 1],
            "tipo": tipos,
            "ventana_inicio": ventana_inicio,
            "ventana_fin": ventana_fin,
            "tiempo_servicio_min": tiempo_servicio,
            "nivel_demanda_base": np.round(nivel_demanda, 2),
        }
    )


def generar_historico_demanda(
    clientes: pd.DataFrame,
    config: Config = Config(),
) -> pd.DataFrame:
    """Histórico diario de pedidos por cliente con estacionalidad y ruido.

    Modelo:
        demanda(c, t) = base(c) * estacional_semana(t) * tendencia_mes(t)
                        * ruido_gauss(c, t) * indicador_pidio(c, t)

    Donde `indicador_pidio` permite que un cliente NO pida ese día con
    probabilidad creciente entre tipos (un minimarket pide casi todos los días,
    un almacén grande pide cada 5–7 días).
    """
    rng = _rng(config.seed + 2)

    fecha_fin = pd.Timestamp("2026-04-30")
    fechas = pd.date_range(end=fecha_fin, periods=config.n_dias_historico, freq="D")
    dias_semana = fechas.dayofweek
    meses = fechas.month

    estacional_semana = np.array([1.10, 1.05, 1.00, 0.95, 0.90, 0.70, 0.60])[dias_semana]
    tendencia_mes = 1.0 + 0.10 * np.sin(2 * np.pi * meses / 12)

    prob_pidio_por_tipo = {
        "minimarket": 0.85,
        "restaurante": 0.65,
        "ferreteria": 0.40,
        "farmacia": 0.55,
        "almacen": 0.20,
    }

    registros = []
    for _, c in clientes.iterrows():
        prob = prob_pidio_por_tipo.get(c["tipo"], 0.5)
        pidio = rng.random(config.n_dias_historico) < prob
        ruido = rng.normal(loc=1.0, scale=RUIDO_DEMANDA, size=config.n_dias_historico)
        ruido = np.clip(ruido, a_min=0.85, a_max=1.15)

        demanda = (
            c["nivel_demanda_base"]
            * estacional_semana
            * tendencia_mes
            * ruido
            * pidio
        )
        demanda = np.where(pidio, np.maximum(demanda, 1.0), 0.0)
        demanda = np.round(demanda, 2)

        for fecha, dem in zip(fechas, demanda):
            registros.append(
                {"fecha": fecha, "cliente_id": c["cliente_id"], "demanda": dem}
            )

    return pd.DataFrame(registros)


def matriz_distancias(coords: np.ndarray) -> np.ndarray:
    """Devuelve la matriz simétrica de distancias en km entre puntos."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2) * GRADOS_A_KM


def construir_dataset(config: Config = Config()) -> dict[str, pd.DataFrame]:
    """Genera y retorna las tres tablas + la matriz de distancias clientes."""
    sucursales = generar_sucursales(config)
    clientes = generar_clientes(config)
    historico = generar_historico_demanda(clientes, config)
    dist_clientes = matriz_distancias(clientes[["lat", "lon"]].to_numpy())
    return {
        "sucursales": sucursales,
        "clientes": clientes,
        "historico_demanda": historico,
        "matriz_distancias_clientes": pd.DataFrame(
            dist_clientes,
            index=clientes["cliente_id"],
            columns=clientes["cliente_id"],
        ),
    }


def guardar_dataset(
    config: Config = Config(),
    destino: Path = DATA_DIR,
) -> dict[str, Path]:
    """Genera el dataset y lo persiste en `data/` como archivos CSV.

    Se elige CSV (en lugar de Parquet) para que cualquier estudiante pueda
    abrir los archivos en Excel/LibreOffice sin instalar dependencias
    adicionales. El tamaño del dataset (~9.000 filas en el histórico) lo
    permite sin penalización de rendimiento.
    """
    destino.mkdir(parents=True, exist_ok=True)
    tablas = construir_dataset(config)
    rutas: dict[str, Path] = {}
    for nombre, df in tablas.items():
        ruta = destino / f"{nombre}.csv"
        df.to_csv(ruta, index=(nombre == "matriz_distancias_clientes"))
        rutas[nombre] = ruta
    return rutas


if __name__ == "__main__":
    rutas = guardar_dataset()
    for nombre, ruta in rutas.items():
        print(f"  {nombre:32s}  →  {ruta}")
