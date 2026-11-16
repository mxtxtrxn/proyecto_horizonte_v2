from __future__ import annotations

import pandas as pd

from horizonte.forecasting import crear_features, ejecutar_forecasting, preparar_fx, split_train_test
from horizonte.generar_datos import generar_fx


def test_preparar_fx_index_y_sin_nulos_en_tipo_cambio():
    df = generar_fx(seed_fx=7, null_ratio=0.05)
    fx = preparar_fx(df)
    assert isinstance(fx.index, pd.DatetimeIndex)
    assert fx.index.freqstr in ("MS", None) or str(fx.index.freq).startswith("<MonthBegin")
    assert fx["tipo_cambio"].isna().sum() == 0
    assert "tasa_interes" in fx.columns and "inflacion" in fx.columns


def test_split_train_test_respeta_corte():
    fx = preparar_fx(generar_fx(seed_fx=7, null_ratio=0.0))
    train, test = split_train_test(fx, corte="2023-01-01")
    assert train.index.max() < pd.Timestamp("2023-01-01")
    assert test.index.min() >= pd.Timestamp("2023-01-01")
    assert len(train) + len(test) == len(fx)


def test_crear_features_genera_lags_y_columnas():
    fx = preparar_fx(generar_fx(seed_fx=7, null_ratio=0.0))
    feat = crear_features(fx)
    for col in ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "mes", "trimestre", "tendencia"]:
        assert col in feat.columns
    assert len(feat) == len(fx) - 12


def test_ejecutar_forecasting_genera_metricas_y_pronostico():
    df = generar_fx(seed_fx=7, null_ratio=0.02)
    res = ejecutar_forecasting(df, corte="2023-01-01", horizonte=6)

    # Métricas
    assert list(res.metricas.index) == ["ARIMA(1,1,1)", "XGBoost/GBM"]
    for col in ["MAE", "RMSE", "MAPE"]:
        assert col in res.metricas.columns
        assert (res.metricas[col] >= 0).all()

    # Pronóstico
    assert list(res.pronostico.columns) == ["fecha", "arima", "xgboost"]
    assert len(res.pronostico) == 6
    assert str(pd.to_datetime(res.pronostico["fecha"].iloc[0]).date()) == "2025-01-01"
    assert str(pd.to_datetime(res.pronostico["fecha"].iloc[-1]).date()) == "2025-06-01"

    # Info de split
    assert res.split_info["test_size"] == 24
    assert res.split_info["train_size"] > 0
