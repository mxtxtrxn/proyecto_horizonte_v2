from __future__ import annotations

import pandas as pd

from horizonte.cleaning import (
    eliminar_duplicados,
    imputar_comentarios,
    imputar_monto_vpg,
    limpiar_leads,
    normalizar_fuente,
    normalizar_sala,
    normalizar_texto,
)
from horizonte.generar_datos import generar_leads


def test_normalizar_texto():
    assert normalizar_texto("  Cancún ") == "cancun"
    assert normalizar_texto("PROVEEDOR A") == "proveedor a"
    assert pd.isna(normalizar_texto(None))


def test_normalizar_fuente_mapea_y_rellena_desconocido():
    df = pd.DataFrame({"fuente": ["Facebook", "fb", "portal web", None, "", "N/A", "otra"]})
    out = normalizar_fuente(df)
    assert out["fuente"].tolist() == ["Social", "Social", "Portal Web", "Desconocido", "Desconocido", "Desconocido", "Desconocido"]


def test_normalizar_sala_mapea_variantes():
    df = pd.DataFrame({"sala": ["Cancún", "cancun", "PDC", "PV", "Cabo", "X"]})
    out = normalizar_sala(df)
    assert out["sala"].tolist() == ["Cancún", "Cancún", "Playa del Carmen", "Puerto Vallarta", "Los Cabos", "Desconocido"]


def test_imputar_comentarios_reemplaza_nulos():
    df = pd.DataFrame({"comentarios": [None, "Hola", pd.NA]})
    out = imputar_comentarios(df)
    assert out["comentarios"].iloc[0] == "Sin comentarios"
    assert out["comentarios"].iloc[1] == "Hola"
    assert out["comentarios"].iloc[2] == "Sin comentarios"


def test_imputar_monto_vpg_solo_en_ventas():
    df = pd.DataFrame({
        "sala": ["Cancún", "Cancún", "Cancún"],
        "fecha_venta": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-10"), None],
        "monto": [100.0, None, None],
        "vpg": [10.0, None, None],
    })
    out = imputar_monto_vpg(df)
    # Segunda fila tiene venta y debe imputarse con mediana de sala (100.0 / 10.0)
    assert out.loc[1, "monto"] == 100.0
    assert out.loc[1, "vpg"] == 10.0
    # Tercera fila sin venta debe seguir nula
    assert pd.isna(out.loc[2, "monto"])
    assert pd.isna(out.loc[2, "vpg"])


def test_eliminar_duplicados():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = eliminar_duplicados(df)
    assert len(out) == 2


def test_limpiar_leads_pipeline_completo():
    raw = generar_leads(n=400, seed_random=42, dup_ratio=0.05, extra_null_ratio=0.1)
    clean = limpiar_leads(raw)
    # Sin duplicados exactos
    assert clean.duplicated().sum() == 0
    # Sin nulos en comentarios
    assert clean["comentarios"].isna().sum() == 0
    # Categorías canónicas
    assert set(clean["fuente"].dropna().unique()).issubset({
        "Proveedor A", "Portal Web", "Social", "Calle", "Exchange", "OTA", "Hotel", "Desconocido"
    })
    assert set(clean["sala"].dropna().unique()).issubset({
        "Cancún", "Playa del Carmen", "Puerto Vallarta", "Los Cabos", "Desconocido"
    })
    # Coherencia: sin venta => sin monto/vpg
    sin_venta = clean["fecha_venta"].isna()
    assert clean.loc[sin_venta, "monto"].isna().all()
    assert clean.loc[sin_venta, "vpg"].isna().all()
    # Ventas con monto/vpg no negativos cuando presentes
    con_venta = clean["fecha_venta"].notna()
    if con_venta.any():
        assert (clean.loc[con_venta, "monto"].dropna() > 0).all()
        assert (clean.loc[con_venta, "vpg"].dropna() > 0).all()
