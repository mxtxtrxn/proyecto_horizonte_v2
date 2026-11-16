from __future__ import annotations

import pandas as pd

from horizonte import config
from horizonte.generar_datos import generar_fx, generar_leads, generar_y_guardar_datos


def test_generar_leads_estructura_y_tipos():
    df = generar_leads(n=200, seed_random=123, dup_ratio=0.05, extra_null_ratio=0.1)
    assert list(df.columns) == config.LEADS_RAW_COLUMNS
    assert len(df) >= 200
    # Debe contener duplicados intencionales
    assert df.duplicated().sum() > 0
    # Columnas esperadas presentes
    assert "lead_id" in df.columns and "fuente" in df.columns


def test_generar_fx_estructura_y_rango():
    df = generar_fx(start="2020-01-01", end="2020-12-01", seed_fx=7, null_ratio=0.0)
    assert list(df.columns) == config.FX_COLUMNS
    assert len(df) == 12
    # Fechas mensuales en formato esperado
    assert df["fecha"].iloc[0] == "2020-01-01"
    assert df["fecha"].iloc[-1] == "2020-12-01"
    assert df["tipo_cambio"].notna().all()


def test_generar_fx_con_nulos_controlados():
    df = generar_fx(start="2020-01-01", end="2021-12-01", seed_fx=7, null_ratio=0.1)
    assert df["tipo_cambio"].isna().sum() >= 1


def test_generar_y_guardar_datos_crea_archivos(tmp_path):
    data_dir = tmp_path / "data"
    leads_path, fx_path = generar_y_guardar_datos(data_dir=data_dir, n_leads=100, seed_random=1, seed_fx=2)
    assert leads_path.exists()
    assert fx_path.exists()

    leads = pd.read_csv(leads_path)
    fx = pd.read_csv(fx_path)
    assert list(leads.columns) == config.LEADS_RAW_COLUMNS
    assert list(fx.columns) == config.FX_COLUMNS
    assert len(leads) >= 100
    assert len(fx) > 0
