"""Forecasting de tipo de cambio MXN/USD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .metrics import calcular_metricas, resumen_metricas

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - depende del entorno
    ARIMA = None
    HAS_STATSMODELS = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover - depende del entorno
    xgb = None
    HAS_XGB = False


CORTE_DEFAULT = "2023-01-01"
FEATURES_DEFAULT = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "tasa_interes", "inflacion", "mes", "trimestre", "tendencia",
]


@dataclass
class ForecastResult:
    metricas: pd.DataFrame
    pronostico: pd.DataFrame
    split_info: dict[str, Any]
    pred_test_arima: pd.Series
    pred_test_ml: pd.Series
    y_test_ml: pd.Series


def preparar_fx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fecha" not in out.columns:
        raise ValueError("El dataset FX debe contener la columna 'fecha'")
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    out = out.dropna(subset=["fecha"]).set_index("fecha").sort_index()
    out = out.asfreq("MS")
    for col in ["tipo_cambio", "tasa_interes", "inflacion"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Interpolación lineal para nulos aislados en tipo_cambio
    if "tipo_cambio" in out.columns:
        out["tipo_cambio"] = out["tipo_cambio"].interpolate(method="linear")
    # Relleno simple para exógenas si quedaran nulos
    for col in ["tasa_interes", "inflacion"]:
        if col in out.columns:
            out[col] = out[col].interpolate(method="linear").ffill().bfill()
    return out


def split_train_test(df_fx: pd.DataFrame, corte: str = CORTE_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    corte_ts = pd.Timestamp(corte)
    train = df_fx[df_fx.index < corte_ts].copy()
    test = df_fx[df_fx.index >= corte_ts].copy()
    if train.empty or test.empty:
        raise ValueError("El split temporal produjo conjuntos vacíos; revisa la fecha de corte y los datos")
    return train, test


def entrenar_arima_y_predecir(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    tc_train = train["tipo_cambio"].astype(float)
    if HAS_STATSMODELS and ARIMA is not None:
        model = ARIMA(tc_train, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.forecast(steps=len(test))
        pred.index = test.index
        return pred.astype(float)
    # Fallback: random walk con deriva
    dy_train = tc_train.diff().dropna()
    deriva = float(dy_train.mean()) if not dy_train.empty else 0.0
    ultimo = float(tc_train.iloc[-1])
    pred_vals = [ultimo + deriva * (i + 1) for i in range(len(test))]
    return pd.Series(pred_vals, index=test.index, dtype=float)


def crear_features(data: pd.DataFrame) -> pd.DataFrame:
    df_f = data.copy()
    for lag in [1, 2, 3, 6, 12]:
        df_f[f"lag_{lag}"] = df_f["tipo_cambio"].shift(lag)
    df_f["mes"] = df_f.index.month
    df_f["trimestre"] = df_f.index.quarter
    df_f["tendencia"] = np.arange(len(df_f))
    return df_f.dropna()


def _crear_modelo_ml():
    if HAS_XGB and xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def entrenar_ml_y_predecir(df_fx: pd.DataFrame, corte: str = CORTE_DEFAULT):
    df_feat = crear_features(df_fx)
    corte_ts = pd.Timestamp(corte)
    tr_feat = df_feat[df_feat.index < corte_ts]
    te_feat = df_feat[df_feat.index >= corte_ts]
    if tr_feat.empty or te_feat.empty:
        raise ValueError("No hay suficientes datos para entrenar/evaluar el modelo ML tras crear features")

    X_train, y_train = tr_feat[FEATURES_DEFAULT], tr_feat["tipo_cambio"]
    X_test, y_test = te_feat[FEATURES_DEFAULT], te_feat["tipo_cambio"]

    modelo = _crear_modelo_ml()
    modelo.fit(X_train, y_train)
    pred = pd.Series(modelo.predict(X_test), index=te_feat.index, dtype=float)
    return modelo, tr_feat, te_feat, X_train, y_train, X_test, y_test, pred


def pronosticar_arima(train: pd.DataFrame, test: pd.DataFrame, horizonte: int = 6):
    fechas_fut = pd.date_range(start="2025-01-01", periods=horizonte, freq="MS")
    tc_train = train["tipo_cambio"].astype(float)
    pred_test = entrenar_arima_y_predecir(train, test)

    if HAS_STATSMODELS and ARIMA is not None:
        model = ARIMA(tc_train, order=(1, 1, 1))
        fit = model.fit()
        fc = fit.get_forecast(steps=horizonte)
        pron = pd.Series(fc.predicted_mean.values, index=fechas_fut, dtype=float)
        ci = fc.conf_int(alpha=0.05)
        ci.index = fechas_fut
        return fechas_fut, pron, ci.iloc[:, 0], ci.iloc[:, 1], pred_test

    # Fallback coherente con notebook
    ultimo = float(test["tipo_cambio"].iloc[-1])
    dy_train = tc_train.diff().dropna()
    deriva = float(dy_train.mean()) if not dy_train.empty else 0.0
    pron = pd.Series([ultimo + deriva * (i + 1) for i in range(horizonte)], index=fechas_fut, dtype=float)
    std_error = float(np.std(test["tipo_cambio"].values - pred_test.values))
    pasos = np.sqrt(np.arange(1, horizonte + 1))
    ci_lo = pron - 1.96 * std_error * pasos
    ci_hi = pron + 1.96 * std_error * pasos
    return fechas_fut, pron, ci_lo, ci_hi, pred_test


def pronosticar_ml(df_fx: pd.DataFrame, modelo, y_test: pd.Series, pred_test: pd.Series, horizonte: int = 6):
    fechas_fut = pd.date_range(start="2025-01-01", periods=horizonte, freq="MS")
    hist = df_fx["tipo_cambio"].astype(float).copy()
    tasa_last = float(df_fx["tasa_interes"].iloc[-1]) if "tasa_interes" in df_fx.columns else 0.0
    infl_last = float(df_fx["inflacion"].iloc[-1]) if "inflacion" in df_fx.columns else 0.0

    pred_fut = []
    for i, fecha in enumerate(fechas_fut):
        row = {
            "lag_1": float(hist.iloc[-1]),
            "lag_2": float(hist.iloc[-2]),
            "lag_3": float(hist.iloc[-3]),
            "lag_6": float(hist.iloc[-6]),
            "lag_12": float(hist.iloc[-12]),
            "tasa_interes": tasa_last,
            "inflacion": infl_last,
            "mes": int(fecha.month),
            "trimestre": int(fecha.quarter),
            "tendencia": int(len(df_fx) + i),
        }
        val = float(modelo.predict(pd.DataFrame([row])[FEATURES_DEFAULT])[0])
        pred_fut.append(val)
        hist = pd.concat([hist, pd.Series([val], index=[fecha])])

    pron = pd.Series(pred_fut, index=fechas_fut, dtype=float)
    residuales = y_test.values - pred_test.values
    std_ml = float(np.std(residuales)) if len(residuales) else 0.0
    ci_lo = pron - 1.96 * std_ml
    ci_hi = pron + 1.96 * std_ml
    return fechas_fut, pron, ci_lo, ci_hi


def ejecutar_forecasting(df_raw: pd.DataFrame, corte: str = CORTE_DEFAULT, horizonte: int = 6) -> ForecastResult:
    df_fx = preparar_fx(df_raw)
    train, test = split_train_test(df_fx, corte=corte)

    # ARIMA (y predicción test)
    fechas_fut_a, pron_a, _, _, pred_test_arima = pronosticar_arima(train, test, horizonte=horizonte)

    # ML
    modelo_ml, _, te_feat, _, _, _, y_test_ml, pred_test_ml = entrenar_ml_y_predecir(df_fx, corte=corte)
    fechas_fut_m, pron_m, _, _, = pronosticar_ml(df_fx, modelo_ml, y_test_ml, pred_test_ml, horizonte=horizonte)

    # Métricas sobre test
    resultados = {
        "ARIMA(1,1,1)": calcular_metricas(test["tipo_cambio"].values, pred_test_arima.reindex(test.index).values),
        "XGBoost/GBM": calcular_metricas(y_test_ml.values, pred_test_ml.reindex(te_feat.index).values),
    }
    metricas_df = resumen_metricas(resultados)

    # Usar mismas fechas futuras (ambos métodos generan 2025-01-01..)
    fechas_fut = fechas_fut_a if len(fechas_fut_a) == horizonte else fechas_fut_m
    pronostico_df = pd.DataFrame(
        {
            "fecha": fechas_fut,
            "arima": np.round(pron_a.reindex(fechas_fut).values, 4),
            "xgboost": np.round(pron_m.reindex(fechas_fut).values, 4),
        }
    )

    split_info = {
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "has_statsmodels": HAS_STATSMODELS,
        "has_xgboost": HAS_XGB,
    }

    return ForecastResult(
        metricas=metricas_df,
        pronostico=pronostico_df,
        split_info=split_info,
        pred_test_arima=pred_test_arima,
        pred_test_ml=pred_test_ml,
        y_test_ml=y_test_ml,
    )
