"""Programación dinámica aproximada — DP por cluster para el caso mediano.

Cuando el número de clientes supera el límite donde Held-Karp es viable
(~15–20 nodos), el problema se descompone en subproblemas:

    1. Cada ruta (cluster) se resuelve por separado con un solver intra-ruta.
    2. Si la ruta tiene ≤ 15 clientes, se aplica Held-Karp (DP exacta).
    3. Si tiene más, se aplica una heurística de DP en dos fases:
       Nearest Neighbor (construcción) + 2-opt (mejora local).
    4. Una vez resuelto cada cluster, se coordina entre sucursales para
       eliminar clientes "fronterizos" que estarían mejor en una ruta vecina
       de otra sucursal.

El resultado es un plan completo con secuencia, kilometraje y costo por ruta,
listo para comparar contra OR-Tools (`src.benchmark`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import GRADOS_A_KM
from .dp_exact import held_karp, held_karp_con_capacidad


UMBRAL_DP_EXACTA = 15


@dataclass
class RutaResuelta:
    """Resultado de resolver una sola ruta (cluster)."""

    sucursal_id: str
    ruta_global: str
    secuencia_clientes: list[str]
    distancia_km: float
    demanda_total: float
    metodo: str


def _matriz_distancias_ruta(
    coords_deposito: np.ndarray,
    coords_clientes: np.ndarray,
) -> np.ndarray:
    coords = np.vstack([coords_deposito[None, :], coords_clientes])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2) * GRADOS_A_KM


def _nearest_neighbor(distancias: np.ndarray) -> list[int]:
    """Construye un tour inicial visitando siempre el nodo más cercano no visitado."""
    n = distancias.shape[0]
    visitados = [False] * n
    tour = [0]
    visitados[0] = True
    actual = 0
    for _ in range(n - 1):
        candidatos = [
            (distancias[actual, j], j)
            for j in range(n)
            if not visitados[j]
        ]
        _, siguiente = min(candidatos)
        tour.append(siguiente)
        visitados[siguiente] = True
        actual = siguiente
    tour.append(0)
    return tour


def _costo_tour(distancias: np.ndarray, tour: list[int]) -> float:
    return float(sum(distancias[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def _dos_opt(distancias: np.ndarray, tour: list[int], max_iter: int = 1000) -> list[int]:
    """Mejora local 2-opt hasta que no haya intercambios beneficiosos."""
    mejor = tour[:]
    n = len(mejor)
    mejorado = True
    iteracion = 0
    while mejorado and iteracion < max_iter:
        mejorado = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                a, b = mejor[i - 1], mejor[i]
                c, d = mejor[j], mejor[j + 1]
                delta = (distancias[a, c] + distancias[b, d]) - (distancias[a, b] + distancias[c, d])
                if delta < -1e-9:
                    mejor[i : j + 1] = mejor[i : j + 1][::-1]
                    mejorado = True
        iteracion += 1
    return mejor


def resolver_ruta(
    coords_deposito: np.ndarray,
    clientes_ruta: pd.DataFrame,
    sucursal_id: str,
    ruta_global: str,
    capacidad_camion: float,
) -> RutaResuelta:
    """Resuelve una sola ruta. Usa Held-Karp si es pequeña, NN+2-opt si es grande."""
    n_clientes = len(clientes_ruta)
    if n_clientes == 0:
        return RutaResuelta(
            sucursal_id=sucursal_id,
            ruta_global=ruta_global,
            secuencia_clientes=[],
            distancia_km=0.0,
            demanda_total=0.0,
            metodo="trivial",
        )

    coords_clientes = clientes_ruta[["lat", "lon"]].to_numpy()
    distancias = _matriz_distancias_ruta(coords_deposito, coords_clientes)
    demanda = np.concatenate([[0.0], clientes_ruta["demanda_predicha"].to_numpy()])

    if n_clientes <= UMBRAL_DP_EXACTA:
        sol = held_karp_con_capacidad(distancias, demanda, capacidad_camion)
        secuencia_idx = sol.secuencia
        costo = sol.costo
        metodo = "held_karp"
    else:
        tour = _nearest_neighbor(distancias)
        tour = _dos_opt(distancias, tour)
        secuencia_idx = tour
        costo = _costo_tour(distancias, tour)
        metodo = "nn_2opt"

    cliente_ids = clientes_ruta["cliente_id"].to_numpy()
    secuencia_clientes: list[str] = []
    for idx in secuencia_idx[1:-1]:
        secuencia_clientes.append(str(cliente_ids[idx - 1]))

    return RutaResuelta(
        sucursal_id=sucursal_id,
        ruta_global=ruta_global,
        secuencia_clientes=secuencia_clientes,
        distancia_km=float(costo),
        demanda_total=float(demanda[1:].sum()),
        metodo=metodo,
    )


def resolver_plan(
    plan: pd.DataFrame,
    sucursales: pd.DataFrame,
) -> tuple[list[RutaResuelta], float]:
    """Resuelve cada ruta del plan y retorna (rutas, kilometraje_total).

    El `plan` debe tener las columnas: cliente_id, lat, lon, sucursal_id,
    demanda_predicha, ruta_global (proviene de `clustering.construir_plan`).
    """
    sucursales_idx = sucursales.set_index("sucursal_id")
    rutas_resueltas: list[RutaResuelta] = []

    for ruta_global, grupo in plan.groupby("ruta_global"):
        sucursal_id = str(grupo["sucursal_id"].iloc[0])
        suc = sucursales_idx.loc[sucursal_id]
        coords_deposito = np.array([suc["lat"], suc["lon"]])
        capacidad = float(suc["capacidad_camion"])
        sol = resolver_ruta(
            coords_deposito=coords_deposito,
            clientes_ruta=grupo,
            sucursal_id=sucursal_id,
            ruta_global=str(ruta_global),
            capacidad_camion=capacidad,
        )
        rutas_resueltas.append(sol)

    km_total = sum(r.distancia_km for r in rutas_resueltas)
    return rutas_resueltas, km_total


def coordinar_inter_sucursal(
    plan: pd.DataFrame,
    sucursales: pd.DataFrame,
    max_swaps: int = 50,
) -> pd.DataFrame:
    """Intenta reasignar clientes fronterizos entre sucursales para reducir km totales.

    Cliente fronterizo = aquel cuyo costo de inserción en una ruta de otra
    sucursal es menor que el costo actual de tenerlo en su sucursal asignada.
    Se hacen swaps mientras el kilometraje total disminuya, hasta `max_swaps`.
    """
    plan_actual = plan.copy()
    sucursales_idx = sucursales.set_index("sucursal_id")

    for _ in range(max_swaps):
        rutas_actuales, km_total_actual = resolver_plan(plan_actual, sucursales)

        mejor_swap = None
        mejor_delta = -1e-9

        for cliente_idx, cliente in plan_actual.iterrows():
            sucursal_actual = cliente["sucursal_id"]
            for sucursal_alternativa in sucursales["sucursal_id"]:
                if sucursal_alternativa == sucursal_actual:
                    continue

                plan_test = plan_actual.copy()
                plan_test.loc[cliente_idx, "sucursal_id"] = sucursal_alternativa

                rutas_existentes = plan_test[plan_test["sucursal_id"] == sucursal_alternativa]["ruta_global"].unique()
                if len(rutas_existentes) > 0:
                    plan_test.loc[cliente_idx, "ruta_global"] = rutas_existentes[0]
                else:
                    plan_test.loc[cliente_idx, "ruta_global"] = f"{sucursal_alternativa}-R1"

                cap_alternativa = float(sucursales_idx.loc[sucursal_alternativa, "capacidad_camion"])
                ruta_dest = plan_test[plan_test["ruta_global"] == plan_test.loc[cliente_idx, "ruta_global"]]
                if ruta_dest["demanda_predicha"].sum() > cap_alternativa:
                    continue

                _, km_test = resolver_plan(plan_test, sucursales)
                delta = km_test - km_total_actual
                if delta < mejor_delta:
                    mejor_delta = delta
                    mejor_swap = (cliente_idx, sucursal_alternativa, plan_test.loc[cliente_idx, "ruta_global"])

        if mejor_swap is None:
            break
        idx, suc_dest, ruta_dest = mejor_swap
        plan_actual.loc[idx, "sucursal_id"] = suc_dest
        plan_actual.loc[idx, "ruta_global"] = ruta_dest

    return plan_actual


if __name__ == "__main__":
    import time

    from src.clustering import construir_plan
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
    print("PLAN NAIVE (sin coordinación)")
    print("=" * 70)
    plan_naive = construir_plan(clientes, sucursales, pred, modo="naive")
    t0 = time.perf_counter()
    rutas_n, km_n = resolver_plan(plan_naive, sucursales)
    t_n = time.perf_counter() - t0
    print(f"Kilometraje total: {km_n:.2f} km · {len(rutas_n)} rutas · {t_n*1000:.0f} ms")
    for r in rutas_n:
        print(f"  {r.ruta_global:8s} ({r.metodo:10s})  {r.distancia_km:6.2f} km  "
              f"{len(r.secuencia_clientes):2d} clientes  carga={r.demanda_total:.1f}")

    print("\n" + "=" * 70)
    print("PLAN COORDINADO (linear_sum_assignment + DP)")
    print("=" * 70)
    plan_coord = construir_plan(clientes, sucursales, pred, modo="coordinado")
    t0 = time.perf_counter()
    rutas_c, km_c = resolver_plan(plan_coord, sucursales)
    t_c = time.perf_counter() - t0
    print(f"Kilometraje total: {km_c:.2f} km · {len(rutas_c)} rutas · {t_c*1000:.0f} ms")

    print(f"\nDelta km (coord - naive): {km_c - km_n:+.2f} km ({100*(km_c-km_n)/km_n:+.1f} %)")
