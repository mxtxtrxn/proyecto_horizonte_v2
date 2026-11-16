from __future__ import annotations

import pandas as pd

from horizonte import config
from horizonte.generar_datos import generar_fx, generar_leads
from horizonte.cleaning import limpiar_leads
from horizonte.validation import validate, validate_fx, validate_leads_clean, validate_leads_raw


def test_validate_leads_raw_ok_con_datos_generados():
    df = generar_leads(n=100, seed_random=42)
    res = validate_leads_raw(df)
    # Puede haber warnings (duplicados, nulos), pero no errores estructurales
    assert res.ok, res.summary()


def test_validate_leads_raw_detecta_columnas_faltantes():
    df = pd.DataFrame({"lead_id": [1], "fuente": ["x"]})
    res = validate_leads_raw(df)
    assert not res.ok
    assert any(i.code == "missing_columns" for i in res.errors)


def test_validate_leads_raw_detecta_valores_no_numericos():
    df = generar_leads(n=50, seed_random=1)
    df.loc[0, "monto"] = "abc"
    res = validate_leads_raw(df)
    assert not res.ok
    assert any(i.code == "non_numeric" for i in res.errors)


def test_validate_leads_clean_ok_tras_limpieza():
    raw = generar_leads(n=300, seed_random=42)
    clean = limpiar_leads(raw)
    res = validate_leads_clean(clean)
    assert res.ok, res.summary()
    assert set(clean["fuente"].unique()).issubset(set(config.FUENTE_CANONICAS))
    assert set(clean["sala"].unique()).issubset(set(config.SALA_CANONICAS))


def test_validate_leads_clean_detecta_categoria_invalida():
    raw = generar_leads(n=100, seed_random=42)
    clean = limpiar_leads(raw)
    clean.loc[0, "fuente"] = "Fuente Invalida"
    res = validate_leads_clean(clean)
    assert not res.ok
    assert any(i.code == "invalid_fuente" for i in res.errors)


def test_validate_fx_ok_y_advertencia_nulos():
    df = generar_fx(seed_fx=7, null_ratio=0.02)
    res = validate_fx(df)
    # nulos en tipo_cambio son warning, no error
    assert res.ok, res.summary()
    assert any(i.code == "nulls" for i in res.warnings)


def test_validate_fx_detecta_fecha_invalida():
    df = generar_fx(seed_fx=7, null_ratio=0.0)
    df.loc[0, "fecha"] = "fecha-mala"
    res = validate_fx(df)
    assert not res.ok
    assert any(i.code == "invalid_dates" for i in res.errors)


def test_validate_dispatch():
    raw = generar_leads(n=50, seed_random=1)
    assert validate("leads", raw).dataset == "leads_raw"
    assert validate("fx", generar_fx(null_ratio=0.0)).dataset == "fx_historico"
