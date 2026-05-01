"""Benchmark de la solución propia contra Google OR-Tools.

Para cerrar el artículo con un sello profesional, la implementación propia
de DP (`src.dp_exact` + `src.dp_approx`) se compara lado a lado con el solver
de Vehicle Routing Problem (VRP) de Google OR-Tools sobre el mismo dataset
sintético. Se reporta:

    - Kilometraje total de cada solución.
    - Tiempo de cómputo.
    - Gap relativo (cuán lejos está la solución propia del óptimo aproximado
      por OR-Tools).
    - Número de rutas de cada solución.

OR-Tools no garantiza el óptimo global del VRP en general (es NP-duro), pero
es un estándar industrial muy refinado y usar su solución como referencia es
suficiente para validar la calidad del motor propio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class SolucionORTools:
    """Resultado del solver OR-Tools."""

    costo_total_km: float
    rutas: list[list[int]] = field(default_factory=list)
    tiempo_cpu_s: float = 0.0
    n_vehiculos_usados: int = 0


def resolver_con_ortools(
    distancias: np.ndarray,
    demanda: np.ndarray,
    capacidad: int | list[int],
    n_vehiculos: int,
    deposito_idx: int = 0,
    tiempo_limite_s: int = 10,
) -> SolucionORTools:
    """Resuelve un CVRP (Capacitated VRP) con OR-Tools.

    Args:
        distancias: matriz simétrica de distancias en km. Se convierte a
            enteros (multiplicado por 1.000) porque OR-Tools opera en enteros.
        demanda: vector de demanda por nodo. `demanda[deposito_idx]` debe ser 0.
        capacidad: capacidad uniforme (int) o por vehículo (lista).
        n_vehiculos: número de vehículos disponibles.
        deposito_idx: índice del depósito en la matriz (default: 0).
        tiempo_limite_s: límite de tiempo del solver en segundos.

    Returns:
        SolucionORTools con costo total (km), rutas como listas de índices y
        tiempo de cómputo.
    """
    import time

    n = distancias.shape[0]
    distancias_int = (distancias * 1000).round().astype(int)
    demanda_int = demanda.round().astype(int)

    if isinstance(capacidad, int):
        capacidades = [capacidad] * n_vehiculos
    else:
        capacidades = list(capacidad)

    manager = pywrapcp.RoutingIndexManager(n, n_vehiculos, deposito_idx)
    routing = pywrapcp.RoutingModel(manager)

    def callback_distancia(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distancias_int[from_node, to_node])

    transit_callback_idx = routing.RegisterTransitCallback(callback_distancia)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    def callback_demanda(from_index):
        return int(demanda_int[manager.IndexToNode(from_index)])

    demand_callback_idx = routing.RegisterUnaryTransitCallback(callback_demanda)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,
        capacidades,
        True,
        "Capacidad",
    )

    parametros = pywrapcp.DefaultRoutingSearchParameters()
    parametros.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametros.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametros.time_limit.FromSeconds(tiempo_limite_s)

    t0 = time.perf_counter()
    solucion = routing.SolveWithParameters(parametros)
    elapsed = time.perf_counter() - t0

    if solucion is None:
        return SolucionORTools(costo_total_km=float("inf"), tiempo_cpu_s=elapsed)

    rutas: list[list[int]] = []
    costo_total_int = 0
    n_usados = 0
    for v in range(n_vehiculos):
        index = routing.Start(v)
        if routing.IsEnd(solucion.Value(routing.NextVar(index))):
            continue
        ruta: list[int] = []
        while not routing.IsEnd(index):
            ruta.append(manager.IndexToNode(index))
            previous = index
            index = solucion.Value(routing.NextVar(index))
            costo_total_int += routing.GetArcCostForVehicle(previous, index, v)
        ruta.append(manager.IndexToNode(index))
        rutas.append(ruta)
        n_usados += 1

    return SolucionORTools(
        costo_total_km=costo_total_int / 1000.0,
        rutas=rutas,
        tiempo_cpu_s=elapsed,
        n_vehiculos_usados=n_usados,
    )


def benchmark_ruta_unica(
    distancias: np.ndarray,
    demanda: np.ndarray,
    capacidad: int,
    sol_dp: dict,
    tiempo_limite_s: int = 5,
) -> pd.DataFrame:
    """Compara la DP propia y OR-Tools sobre un único depósito y un vehículo.

    Args:
        distancias, demanda, capacidad: input del problema.
        sol_dp: dict con campos `costo` y `tiempo_s` de la solución propia.
        tiempo_limite_s: límite OR-Tools.

    Returns:
        DataFrame de una fila con costos, tiempos y gap.
    """
    sol_or = resolver_con_ortools(
        distancias=distancias,
        demanda=demanda,
        capacidad=capacidad,
        n_vehiculos=1,
        tiempo_limite_s=tiempo_limite_s,
    )
    gap = (sol_dp["costo"] - sol_or.costo_total_km) / sol_or.costo_total_km * 100 if sol_or.costo_total_km > 0 else 0.0
    return pd.DataFrame(
        [
            {
                "metodo": "DP propia",
                "costo_km": round(sol_dp["costo"], 2),
                "tiempo_s": round(sol_dp.get("tiempo_s", 0.0), 4),
                "gap_vs_ortools_pct": round(gap, 2),
            },
            {
                "metodo": "OR-Tools",
                "costo_km": round(sol_or.costo_total_km, 2),
                "tiempo_s": round(sol_or.tiempo_cpu_s, 4),
                "gap_vs_ortools_pct": 0.0,
            },
        ]
    )


def benchmark_plan_completo(
    plan: pd.DataFrame,
    sucursales: pd.DataFrame,
    rutas_propias: list,
    km_propio_total: float,
    tiempo_propio_s: float,
    tiempo_limite_s: int = 30,
) -> pd.DataFrame:
    """Compara el plan completo (motor propio) contra OR-Tools sobre el mismo dataset.

    Para que la comparación sea justa, OR-Tools resuelve el problema con un
    único depósito artificial (centro geométrico de las sucursales) y la flota
    consolidada de toda la empresa. La solución propia ya tiene asignación
    a sucursales y ruteo por sucursal.
    """
    import time

    from .data import GRADOS_A_KM

    centro_lat = sucursales["lat"].mean()
    centro_lon = sucursales["lon"].mean()

    coords = np.vstack(
        [
            [[centro_lat, centro_lon]],
            plan[["lat", "lon"]].to_numpy(),
        ]
    )
    diff = coords[:, None, :] - coords[None, :, :]
    distancias = np.linalg.norm(diff, axis=2) * GRADOS_A_KM
    demanda = np.concatenate([[0.0], plan["demanda_predicha"].to_numpy()])

    n_vehiculos = int(sucursales["n_camiones"].sum())
    capacidad_min = int(sucursales["capacidad_camion"].min())

    sol_or = resolver_con_ortools(
        distancias=distancias,
        demanda=demanda,
        capacidad=capacidad_min,
        n_vehiculos=n_vehiculos,
        tiempo_limite_s=tiempo_limite_s,
    )

    if sol_or.costo_total_km == float("inf"):
        gap = float("nan")
    else:
        gap = (km_propio_total - sol_or.costo_total_km) / sol_or.costo_total_km * 100

    return pd.DataFrame(
        [
            {
                "metodo": "Motor propio (DP + ML)",
                "km_total": round(km_propio_total, 2),
                "n_rutas": len(rutas_propias),
                "tiempo_s": round(tiempo_propio_s, 4),
                "gap_vs_ortools_pct": round(gap, 2),
            },
            {
                "metodo": "OR-Tools (CVRP)",
                "km_total": round(sol_or.costo_total_km, 2),
                "n_rutas": sol_or.n_vehiculos_usados,
                "tiempo_s": round(sol_or.tiempo_cpu_s, 4),
                "gap_vs_ortools_pct": 0.0,
            },
        ]
    )


if __name__ == "__main__":
    import time

    from src.clustering import construir_plan
    from src.data import construir_dataset, GRADOS_A_KM
    from src.demand import entrenar, predecir_dia
    from src.dp_approx import resolver_plan
    from src.dp_exact import held_karp

    tablas = construir_dataset()
    sucursales = tablas["sucursales"]
    clientes = tablas["clientes"]
    historico = tablas["historico_demanda"]

    print("=" * 70)
    print("BENCHMARK 1 — TSP pequeño: 12 clientes, una sucursal")
    print("=" * 70)
    clientes_chico = clientes.head(12)
    sucursal = sucursales.iloc[0]
    coords = np.vstack(
        [
            [[sucursal["lat"], sucursal["lon"]]],
            clientes_chico[["lat", "lon"]].to_numpy(),
        ]
    )
    distancias_chico = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) * GRADOS_A_KM
    demanda_chico = np.concatenate([[0.0], clientes_chico["nivel_demanda_base"].to_numpy()])

    t0 = time.perf_counter()
    sol_dp = held_karp(distancias_chico)
    t_dp = time.perf_counter() - t0
    tabla1 = benchmark_ruta_unica(
        distancias=distancias_chico,
        demanda=demanda_chico,
        capacidad=2000,
        sol_dp={"costo": sol_dp.costo, "tiempo_s": t_dp},
    )
    print(tabla1.to_string(index=False))

    print("\n" + "=" * 70)
    print("BENCHMARK 2 — Plan completo: 50 clientes, 5 sucursales")
    print("=" * 70)
    modelo_dem = entrenar(historico, clientes)
    fecha_obj = pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1)
    pred = predecir_dia(modelo_dem, fecha_obj, historico, clientes)
    plan = construir_plan(clientes, sucursales, pred, modo="coordinado")

    t0 = time.perf_counter()
    rutas, km_total = resolver_plan(plan, sucursales)
    t_propio = time.perf_counter() - t0

    tabla2 = benchmark_plan_completo(
        plan=plan,
        sucursales=sucursales,
        rutas_propias=rutas,
        km_propio_total=km_total,
        tiempo_propio_s=t_propio,
        tiempo_limite_s=15,
    )
    print(tabla2.to_string(index=False))
