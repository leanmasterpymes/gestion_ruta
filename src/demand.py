"""Predicción de demanda por cliente con LightGBM.

Modelo de regresión que estima la demanda de cada cliente para un día objetivo,
usando features temporales (día de la semana, mes), de lag (demanda hace 1, 7
y 14 días) y de rolling mean (medias móviles de 7 y 30 días). El error se
reporta en MAE y MAPE para alimentar el cálculo de capacidad por camión.

Estrategia de validación:
    Split temporal — los últimos 30 días del histórico se reservan como test;
    el resto se usa para entrenar. Esto evita la fuga de información típica
    cuando se hace un split aleatorio sobre series temporales.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


FEATURES_TEMPORALES = ["dayofweek", "month", "day", "weekofyear"]
FEATURES_LAG = ["lag_1", "lag_7", "lag_14"]
FEATURES_ROLLING = ["roll_mean_7", "roll_mean_30"]
FEATURES_CLIENTE = ["tipo", "ventana_inicio", "ventana_fin", "tiempo_servicio_min", "nivel_demanda_base"]
TARGET = "demanda"
DIAS_TEST = 30


@dataclass
class ModeloDemanda:
    """Empaqueta el regresor entrenado, las columnas usadas y las métricas."""

    modelo: lgb.LGBMRegressor
    features: list[str]
    categoricas: list[str]
    metricas: dict[str, float]


def construir_features(
    historico: pd.DataFrame,
    clientes: pd.DataFrame,
) -> pd.DataFrame:
    """Genera la tabla con features temporales, de lag y de cliente.

    El histórico de entrada debe tener columnas: fecha, cliente_id, demanda.
    """
    df = historico.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["cliente_id", "fecha"]).reset_index(drop=True)

    df["dayofweek"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["day"] = df["fecha"].dt.day
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype(int)

    grupo = df.groupby("cliente_id")["demanda"]
    df["lag_1"] = grupo.shift(1)
    df["lag_7"] = grupo.shift(7)
    df["lag_14"] = grupo.shift(14)
    df["roll_mean_7"] = grupo.shift(1).rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["roll_mean_30"] = grupo.shift(1).rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

    df = df.merge(clientes[["cliente_id", *FEATURES_CLIENTE]], on="cliente_id", how="left")
    df["tipo"] = df["tipo"].astype("category")

    return df


def _split_temporal(df: pd.DataFrame, dias_test: int = DIAS_TEST) -> tuple[pd.DataFrame, pd.DataFrame]:
    fecha_corte = df["fecha"].max() - pd.Timedelta(days=dias_test)
    train = df[df["fecha"] <= fecha_corte].copy()
    test = df[df["fecha"] > fecha_corte].copy()
    return train, test


def _mape_seguro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE que ignora días con demanda real cero (cliente no pidió)."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE (Weighted Absolute Percentage Error) — métrica estándar en retail.

    A diferencia del MAPE, no se infla con clientes de baja demanda y refleja
    el error agregado ponderado por volumen, que es lo que importa en planificación
    de capacidad. Se calcula sobre los días con pedido real (demanda > 0).
    """
    mask = y_true > 0
    total = float(y_true[mask].sum())
    if total == 0:
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).sum() / total * 100)


def entrenar(
    historico: pd.DataFrame,
    clientes: pd.DataFrame,
    dias_test: int = DIAS_TEST,
    seed: int = 42,
) -> ModeloDemanda:
    """Entrena el regresor con split temporal y devuelve el modelo y métricas."""
    df = construir_features(historico, clientes)
    df = df.dropna(subset=FEATURES_LAG + FEATURES_ROLLING).reset_index(drop=True)

    features = FEATURES_TEMPORALES + FEATURES_LAG + FEATURES_ROLLING + FEATURES_CLIENTE
    categoricas = ["tipo"]

    train, test = _split_temporal(df, dias_test=dias_test)
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    modelo = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=seed,
        verbose=-1,
    )
    modelo.fit(X_train, y_train, categorical_feature=categoricas)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    test_eval = test.copy()
    test_eval["pred"] = pred_test
    sucursal_dia = (
        test_eval.assign(fecha=test_eval["fecha"].dt.date)
        .groupby(["fecha"], as_index=False)[["demanda", "pred"]]
        .sum()
    )
    wape_total = _wape(sucursal_dia["demanda"].to_numpy(), sucursal_dia["pred"].to_numpy())

    metricas = {
        "mae_train": float(mean_absolute_error(y_train, pred_train)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "mape_test_cliente": _mape_seguro(y_test.to_numpy(), pred_test),
        "wape_test_cliente": _wape(y_test.to_numpy(), pred_test),
        "wape_test_dia": wape_total,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    return ModeloDemanda(modelo=modelo, features=features, categoricas=categoricas, metricas=metricas)


def predecir_dia(
    modelo_dem: ModeloDemanda,
    fecha: pd.Timestamp,
    historico: pd.DataFrame,
    clientes: pd.DataFrame,
) -> pd.DataFrame:
    """Devuelve la demanda esperada para `fecha`, por cliente.

    Usa el histórico hasta el día anterior a `fecha` para calcular los lags.
    """
    fecha = pd.Timestamp(fecha)
    historico_extendido = historico.copy()
    historico_extendido["fecha"] = pd.to_datetime(historico_extendido["fecha"])

    placeholder = pd.DataFrame(
        {
            "fecha": [fecha] * len(clientes),
            "cliente_id": clientes["cliente_id"].to_list(),
            "demanda": [np.nan] * len(clientes),
        }
    )
    extendido = pd.concat([historico_extendido, placeholder], ignore_index=True)

    df = construir_features(extendido, clientes)
    df_fecha = df[df["fecha"] == fecha].copy()
    pred = modelo_dem.modelo.predict(df_fecha[modelo_dem.features])
    pred = np.maximum(pred, 0.0)

    return pd.DataFrame(
        {
            "cliente_id": df_fecha["cliente_id"].to_numpy(),
            "fecha": fecha,
            "demanda_predicha": np.round(pred, 2),
        }
    )


if __name__ == "__main__":
    from src.data import construir_dataset

    tablas = construir_dataset()
    clientes = tablas["clientes"]
    historico = tablas["historico_demanda"]

    modelo_dem = entrenar(historico, clientes)
    print("Métricas de validación temporal:")
    for k, v in modelo_dem.metricas.items():
        print(f"  {k:12s} = {v:.4f}" if isinstance(v, float) else f"  {k:12s} = {v}")

    fecha_objetivo = pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1)
    pred = predecir_dia(modelo_dem, fecha_objetivo, historico, clientes)
    print(f"\nPredicción para {fecha_objetivo.date()} — primeros 5 clientes:")
    print(pred.head())
    print(f"\nDemanda total esperada: {pred['demanda_predicha'].sum():.2f} unidades")
