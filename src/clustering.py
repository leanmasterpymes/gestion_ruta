"""Asignación de clientes a sucursales + clustering de rutas con capacidad.

Provee dos modos de asignación cliente → sucursal para que el artículo muestre
el contraste antes/después:

1.  **`asignar_sucursal_naive`** — cada sucursal toma sus clientes más cercanos
    sin coordinar con las demás. Es el patrón actual de "sucursales como islas"
    que produce solapamientos: dos camiones de la misma empresa pasando a tres
    cuadras del mismo cliente.

2.  **`asignar_sucursal_coordinada`** — asignación coordinada vía problema de
    asignación lineal (Hungarian / `linear_sum_assignment`) que respeta la
    capacidad total de cada sucursal y minimiza la distancia agregada
    cliente-sucursal. Elimina el solapamiento.

Una vez resuelta la asignación inter-sucursal, dentro de cada sucursal se
agrupan los clientes en rutas (clusters) que respeten la capacidad de un
camión, vía K-Means con post-procesamiento de balanceo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .data import GRADOS_A_KM


def _distancias_clientes_sucursales(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
) -> np.ndarray:
    """Matriz (n_clientes, n_sucursales) de distancias en km."""
    coords_c = clientes[["lat", "lon"]].to_numpy()
    coords_s = sucursales[["lat", "lon"]].to_numpy()
    diff = coords_c[:, None, :] - coords_s[None, :, :]
    return np.linalg.norm(diff, axis=2) * GRADOS_A_KM


def asignar_sucursal_naive(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
) -> pd.Series:
    """Asigna cada cliente a la sucursal geográficamente más cercana.

    No considera capacidad de la flota ni balance entre sucursales — es la
    asignación "isla" que el artículo critica.
    """
    distancias = _distancias_clientes_sucursales(clientes, sucursales)
    idx_sucursal = distancias.argmin(axis=1)
    return pd.Series(
        sucursales["sucursal_id"].to_numpy()[idx_sucursal],
        index=clientes["cliente_id"].to_numpy(),
        name="sucursal_id",
    )


def asignar_sucursal_coordinada(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
    demanda_estimada: pd.Series,
) -> pd.Series:
    """Asigna clientes a sucursales respetando capacidad total de la flota.

    Modelo: asignación lineal sobre una matriz expandida donde cada sucursal
    aparece replicada según el número de "slots" de capacidad que ofrece. Cada
    slot equivale a una unidad de demanda esperada. El cliente se asigna al
    slot que minimiza distancia, garantizando que ninguna sucursal excede su
    capacidad agregada (n_camiones × capacidad_camion).

    Para que la matriz sea tratable, los slots se discretizan en bloques del
    tamaño promedio de demanda por cliente; la asignación final preserva la
    integridad cliente-sucursal.
    """
    distancias = _distancias_clientes_sucursales(clientes, sucursales)
    n_clientes = len(clientes)
    n_sucursales = len(sucursales)

    capacidades = (sucursales["n_camiones"] * sucursales["capacidad_camion"]).to_numpy()
    demanda_por_cliente = demanda_estimada.reindex(clientes["cliente_id"]).fillna(0).to_numpy()
    demanda_total = demanda_por_cliente.sum()
    factor = max(1.0, demanda_total / capacidades.sum())
    capacidades_efectivas = (capacidades * factor).astype(int)

    slots_por_sucursal = []
    coste_por_slot = []
    for s in range(n_sucursales):
        n_slots = max(1, int(np.ceil(capacidades_efectivas[s] / max(1, demanda_por_cliente.mean()))))
        slots_por_sucursal.append(n_slots)
        coste_por_slot.extend([distancias[:, s]] * n_slots)

    while sum(slots_por_sucursal) < n_clientes:
        s = int(np.argmax(capacidades_efectivas))
        slots_por_sucursal[s] += 1
        coste_por_slot.append(distancias[:, s])

    matriz_coste = np.column_stack(coste_por_slot)
    fila_idx, col_idx = linear_sum_assignment(matriz_coste)

    slot_a_sucursal: list[str] = []
    for s, n_slots in enumerate(slots_por_sucursal):
        slot_a_sucursal.extend([sucursales["sucursal_id"].iloc[s]] * n_slots)

    asignacion = pd.Series(
        [slot_a_sucursal[c] for c in col_idx],
        index=clientes["cliente_id"].iloc[fila_idx].to_numpy(),
        name="sucursal_id",
    )
    return asignacion.reindex(clientes["cliente_id"].to_numpy())


def agrupar_rutas_dentro_sucursal(
    clientes_sucursal: pd.DataFrame,
    demanda_estimada: pd.Series,
    capacidad_camion: float,
    seed: int = 42,
) -> pd.Series:
    """Agrupa los clientes de una sucursal en rutas (clusters) que respeten la capacidad.

    Estrategia:
        1. Calcular `k = ceil(demanda_total / capacidad_camion)` como número
           mínimo de rutas necesarias.
        2. Ejecutar K-Means sobre coordenadas (lat, lon).
        3. Si algún cluster excede capacidad, mover el cliente más lejano del
           centroide a un cluster vecino con holgura. Repetir hasta cumplir.
    """
    if len(clientes_sucursal) == 0:
        return pd.Series([], dtype=object, name="ruta_id")

    demanda = demanda_estimada.reindex(clientes_sucursal["cliente_id"]).fillna(0).to_numpy()
    demanda_total = demanda.sum()
    k = max(1, int(np.ceil(demanda_total / capacidad_camion)))
    k = min(k, len(clientes_sucursal))

    coords = clientes_sucursal[["lat", "lon"]].to_numpy()
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(coords)

    labels = _balancear_capacidad(labels, demanda, coords, km.cluster_centers_, capacidad_camion)

    ids_ruta = [f"R{i+1}" for i in labels]
    return pd.Series(ids_ruta, index=clientes_sucursal["cliente_id"].to_numpy(), name="ruta_id")


def _balancear_capacidad(
    labels: np.ndarray,
    demanda: np.ndarray,
    coords: np.ndarray,
    centroides: np.ndarray,
    capacidad: float,
    max_iter: int = 50,
) -> np.ndarray:
    """Mueve clientes entre clusters fronterizos para respetar la capacidad."""
    labels = labels.copy()
    for _ in range(max_iter):
        cargas = np.array([demanda[labels == k].sum() for k in range(len(centroides))])
        excedidos = np.where(cargas > capacidad)[0]
        if len(excedidos) == 0:
            return labels
        k_origen = int(excedidos[np.argmax(cargas[excedidos])])
        miembros = np.where(labels == k_origen)[0]
        dist_centro = np.linalg.norm(coords[miembros] - centroides[k_origen], axis=1)
        cliente = int(miembros[np.argmax(dist_centro)])

        candidatos = np.argsort(np.linalg.norm(centroides - coords[cliente], axis=1))
        movido = False
        for k_destino in candidatos:
            if k_destino == k_origen:
                continue
            if cargas[k_destino] + demanda[cliente] <= capacidad:
                labels[cliente] = k_destino
                movido = True
                break
        if not movido:
            return labels
    return labels


def construir_plan(
    clientes: pd.DataFrame,
    sucursales: pd.DataFrame,
    demanda_predicha: pd.DataFrame,
    modo: str = "coordinado",
) -> pd.DataFrame:
    """Devuelve la tabla cliente → sucursal → ruta lista para el motor de DP.

    Args:
        clientes, sucursales: tablas del dataset.
        demanda_predicha: DataFrame con columnas `cliente_id` y `demanda_predicha`.
        modo: "naive" (sin coordinación) o "coordinado" (con linear_sum_assignment).
    """
    demanda_serie = demanda_predicha.set_index("cliente_id")["demanda_predicha"]

    if modo == "naive":
        sucursal_de = asignar_sucursal_naive(clientes, sucursales)
    elif modo == "coordinado":
        sucursal_de = asignar_sucursal_coordinada(clientes, sucursales, demanda_serie)
    else:
        raise ValueError(f"Modo desconocido: {modo!r}")

    plan = clientes[["cliente_id", "lat", "lon"]].copy()
    plan["sucursal_id"] = plan["cliente_id"].map(sucursal_de)
    plan["demanda_predicha"] = plan["cliente_id"].map(demanda_serie).fillna(0)

    rutas: list[pd.Series] = []
    for sucursal_id, grupo in plan.groupby("sucursal_id"):
        capacidad = float(sucursales.set_index("sucursal_id").loc[sucursal_id, "capacidad_camion"])
        ruta_ids = agrupar_rutas_dentro_sucursal(grupo, demanda_serie, capacidad)
        rutas.append(ruta_ids.rename(f"ruta_{sucursal_id}"))

    plan["ruta_id"] = pd.concat(rutas).reindex(plan["cliente_id"].to_numpy()).to_numpy()
    plan["ruta_global"] = plan["sucursal_id"].astype(str) + "-" + plan["ruta_id"].astype(str)

    return plan


if __name__ == "__main__":
    from src.data import construir_dataset
    from src.demand import entrenar, predecir_dia

    tablas = construir_dataset()
    sucursales = tablas["sucursales"]
    clientes = tablas["clientes"]
    historico = tablas["historico_demanda"]

    modelo_dem = entrenar(historico, clientes)
    fecha_obj = pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1)
    pred = predecir_dia(modelo_dem, fecha_obj, historico, clientes)

    print("=" * 70)
    print("ASIGNACIÓN NAIVE (cada sucursal toma sus clientes más cercanos)")
    print("=" * 70)
    plan_naive = construir_plan(clientes, sucursales, pred, modo="naive")
    resumen_n = plan_naive.groupby("sucursal_id").agg(
        n_clientes=("cliente_id", "count"),
        demanda_total=("demanda_predicha", "sum"),
        n_rutas=("ruta_global", "nunique"),
    )
    print(resumen_n)

    print("\n" + "=" * 70)
    print("ASIGNACIÓN COORDINADA (linear_sum_assignment, respeta capacidad)")
    print("=" * 70)
    plan_coord = construir_plan(clientes, sucursales, pred, modo="coordinado")
    resumen_c = plan_coord.groupby("sucursal_id").agg(
        n_clientes=("cliente_id", "count"),
        demanda_total=("demanda_predicha", "sum"),
        n_rutas=("ruta_global", "nunique"),
    )
    print(resumen_c)

    cap_total = sucursales.set_index("sucursal_id").apply(
        lambda r: r["n_camiones"] * r["capacidad_camion"], axis=1
    )
    print("\nCapacidad total por sucursal (n_camiones × capacidad_camion):")
    print(cap_total)
