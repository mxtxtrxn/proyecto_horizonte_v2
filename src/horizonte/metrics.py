"""Cálculo de métricas de evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calcular_metricas(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    # Evitar división por cero
    denom = np.where(y_true_arr == 0, np.nan, y_true_arr)
    mape = float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def resumen_metricas(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(resultados, orient="index")
    # Orden de columnas consistente
    cols = [c for c in ["MAE", "RMSE", "MAPE"] if c in df.columns]
    return df[cols].round(4)
