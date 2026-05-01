"""Programación dinámica exacta — Held-Karp para el caso pequeño.

Implementación didáctica del algoritmo de Bellman-Held-Karp (1962) para el
TSP, con complejidad O(2^n · n^2) en tiempo y O(2^n · n) en espacio. Se aplica
al caso pequeño del artículo (1 sucursal, 10–15 clientes) para enseñar
explícitamente la idea de subproblemas con bitmask y memoización.

Recurrencia:
    g(S, i) = min sobre j en S\\{i} de [ g(S\\{i}, j) + d(j, i) ]

Donde S es el subconjunto de nodos visitados (representado como entero con
bits), i es el último nodo visitado y d(j, i) es la distancia entre j e i.
La solución del TSP cerrado es:
    OPT = min sobre i de [ g({1,...,n-1}, i) + d(i, 0) ]

La variante con capacidad agrega un filtro que descarta los subconjuntos S
cuya demanda acumulada exceda la capacidad del vehículo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolucionDP:
    """Resultado del solver Held-Karp."""

    costo: float
    secuencia: list[int]
    n_nodos: int
    n_subproblemas: int


def held_karp(distancias: np.ndarray) -> SolucionDP:
    """Resuelve el TSP exacto con DP sobre subconjuntos (bitmask).

    Args:
        distancias: matriz simétrica de distancias entre nodos. El nodo 0 es
            siempre el depósito (sucursal). El recorrido empieza y termina
            en el nodo 0.

    Returns:
        SolucionDP con costo óptimo, secuencia de nodos visitados (incluye
        depósito al inicio y al final) y conteo de subproblemas evaluados.

    Raises:
        ValueError: si la matriz no es cuadrada o tiene más de 22 nodos
            (límite práctico para 64-bit bitmask y memoria razonable).
    """
    n = distancias.shape[0]
    if distancias.shape[0] != distancias.shape[1]:
        raise ValueError("La matriz de distancias debe ser cuadrada.")
    if n > 22:
        raise ValueError(
            f"Held-Karp exacto no es viable para n={n} nodos. "
            "Use src.dp_approx para n > ~20."
        )
    if n < 2:
        return SolucionDP(costo=0.0, secuencia=[0], n_nodos=n, n_subproblemas=0)

    INF = float("inf")
    dp: dict[tuple[int, int], tuple[float, int]] = {}

    for i in range(1, n):
        dp[(1 << i, i)] = (distancias[0, i], 0)

    for tam_subset in range(2, n):
        for subset in _subsets_de_tamano(n - 1, tam_subset):
            for i in range(1, n):
                bit_i = 1 << i
                if not (subset & bit_i):
                    continue
                subset_prev = subset & ~bit_i
                mejor_costo = INF
                mejor_prev = -1
                for j in range(1, n):
                    bit_j = 1 << j
                    if j == i or not (subset_prev & bit_j):
                        continue
                    candidato = dp.get((subset_prev, j), (INF, -1))[0] + distancias[j, i]
                    if candidato < mejor_costo:
                        mejor_costo = candidato
                        mejor_prev = j
                dp[(subset, i)] = (mejor_costo, mejor_prev)

    subset_final = (1 << n) - 2
    mejor_costo = INF
    ultimo = -1
    for i in range(1, n):
        candidato = dp.get((subset_final, i), (INF, -1))[0] + distancias[i, 0]
        if candidato < mejor_costo:
            mejor_costo = candidato
            ultimo = i

    secuencia = _reconstruir_ruta(dp, subset_final, ultimo, n)
    return SolucionDP(
        costo=float(mejor_costo),
        secuencia=secuencia,
        n_nodos=n,
        n_subproblemas=len(dp),
    )


def held_karp_con_capacidad(
    distancias: np.ndarray,
    demanda: np.ndarray,
    capacidad: float,
) -> SolucionDP:
    """Variante con restricción de capacidad acumulada del vehículo.

    Args:
        distancias: matriz simétrica de distancias.
        demanda: vector de demanda por nodo. `demanda[0]` se ignora (depósito).
        capacidad: capacidad máxima del vehículo.

    Returns:
        SolucionDP con la mejor ruta que visita todos los nodos sin exceder
        la capacidad acumulada en ningún momento.

    Raises:
        ValueError: si la demanda total excede la capacidad (problema infactible).
    """
    n = distancias.shape[0]
    if n > 22:
        raise ValueError(f"Held-Karp exacto no es viable para n={n}. Use src.dp_approx.")
    demanda_total = float(demanda[1:].sum())
    if demanda_total > capacidad:
        raise ValueError(
            f"Demanda total ({demanda_total:.1f}) excede capacidad ({capacidad:.1f}). "
            "Divida los clientes en más rutas."
        )

    INF = float("inf")
    dp: dict[tuple[int, int], tuple[float, int]] = {}

    for i in range(1, n):
        if demanda[i] <= capacidad:
            dp[(1 << i, i)] = (distancias[0, i], 0)

    for tam_subset in range(2, n):
        for subset in _subsets_de_tamano(n - 1, tam_subset):
            carga = sum(demanda[k] for k in range(1, n) if subset & (1 << k))
            if carga > capacidad:
                continue
            for i in range(1, n):
                bit_i = 1 << i
                if not (subset & bit_i):
                    continue
                subset_prev = subset & ~bit_i
                mejor_costo = INF
                mejor_prev = -1
                for j in range(1, n):
                    bit_j = 1 << j
                    if j == i or not (subset_prev & bit_j):
                        continue
                    candidato = dp.get((subset_prev, j), (INF, -1))[0] + distancias[j, i]
                    if candidato < mejor_costo:
                        mejor_costo = candidato
                        mejor_prev = j
                if mejor_costo < INF:
                    dp[(subset, i)] = (mejor_costo, mejor_prev)

    subset_final = (1 << n) - 2
    mejor_costo = INF
    ultimo = -1
    for i in range(1, n):
        candidato = dp.get((subset_final, i), (INF, -1))[0] + distancias[i, 0]
        if candidato < mejor_costo:
            mejor_costo = candidato
            ultimo = i

    secuencia = _reconstruir_ruta(dp, subset_final, ultimo, n)
    return SolucionDP(
        costo=float(mejor_costo),
        secuencia=secuencia,
        n_nodos=n,
        n_subproblemas=len(dp),
    )


def _subsets_de_tamano(n_no_deposito: int, tam: int):
    """Genera todos los subsets de tamaño `tam` sobre `n_no_deposito` nodos.

    Los subsets se representan como enteros con bits puestos en las posiciones
    1..n_no_deposito (la posición 0 se reserva al depósito).
    """
    nodos = list(range(1, n_no_deposito + 1))
    yield from _combinaciones_bitmask(nodos, tam)


def _combinaciones_bitmask(nodos: list[int], tam: int):
    if tam == 0:
        yield 0
        return
    if tam > len(nodos):
        return
    cabeza, cola = nodos[0], nodos[1:]
    for sub in _combinaciones_bitmask(cola, tam - 1):
        yield (1 << cabeza) | sub
    yield from _combinaciones_bitmask(cola, tam)


def _reconstruir_ruta(
    dp: dict[tuple[int, int], tuple[float, int]],
    subset_final: int,
    ultimo: int,
    n: int,
) -> list[int]:
    """Reconstruye la secuencia óptima desde el depósito y de vuelta."""
    secuencia: list[int] = [0]
    subset = subset_final
    actual = ultimo
    pila: list[int] = []
    while actual != 0:
        pila.append(actual)
        prev = dp[(subset, actual)][1]
        subset &= ~(1 << actual)
        actual = prev
    secuencia.extend(reversed(pila))
    secuencia.append(0)
    return secuencia


if __name__ == "__main__":
    import time

    from src.data import construir_dataset, GRADOS_A_KM

    tablas = construir_dataset()
    clientes = tablas["clientes"].head(12)
    sucursal = tablas["sucursales"].iloc[0]

    coords_nodos = np.vstack(
        [
            [[sucursal["lat"], sucursal["lon"]]],
            clientes[["lat", "lon"]].to_numpy(),
        ]
    )
    diff = coords_nodos[:, None, :] - coords_nodos[None, :, :]
    distancias = np.linalg.norm(diff, axis=2) * GRADOS_A_KM

    print(f"Resolviendo TSP exacto sobre {len(coords_nodos)} nodos (depósito + 12 clientes)...")
    t0 = time.perf_counter()
    sol = held_karp(distancias)
    elapsed = time.perf_counter() - t0
    print(f"  Costo óptimo:     {sol.costo:.2f} km")
    print(f"  Secuencia:        {sol.secuencia}")
    print(f"  Subproblemas:     {sol.n_subproblemas}")
    print(f"  Tiempo cómputo:   {elapsed*1000:.1f} ms")

    demanda = np.array([0.0] + clientes["nivel_demanda_base"].tolist())
    print(f"\nResolviendo con capacidad = 1500 (demanda total = {demanda.sum():.1f})...")
    sol_cap = held_karp_con_capacidad(distancias, demanda, capacidad=1500.0)
    print(f"  Costo óptimo:     {sol_cap.costo:.2f} km")
    print(f"  Secuencia:        {sol_cap.secuencia}")
    print(f"  Subproblemas:     {sol_cap.n_subproblemas}")
