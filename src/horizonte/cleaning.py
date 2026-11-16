"""Limpieza y normalización de datos de leads."""

from __future__ import annotations

import unicodedata

import pandas as pd

from . import config


def normalizar_texto(s: object) -> object:
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s


def normalizar_fuente(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fuente" not in out.columns:
        return out
    fuente_norm = out["fuente"].apply(normalizar_texto).map(config.FUENTE_MAP)
    out["fuente"] = fuente_norm.fillna("Desconocido")
    return out


def normalizar_sala(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sala" not in out.columns:
        return out
    sala_norm = out["sala"].apply(normalizar_texto).map(config.SALA_MAP)
    out["sala"] = sala_norm.fillna("Desconocido")
    return out


def imputar_comentarios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "comentarios" in out.columns:
        out["comentarios"] = out["comentarios"].fillna("Sin comentarios")
    return out


def imputar_monto_vpg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"fecha_venta", "monto", "vpg", "sala"}
    if not required.issubset(out.columns):
        return out

    fecha_venta = pd.to_datetime(out["fecha_venta"], errors="coerce")
    mask_venta = fecha_venta.notna()

    # Medianas por sala usando solo ventas con valores válidos
    ventas_validas = out.loc[mask_venta].copy()
    medianas_monto = ventas_validas.groupby("sala")["monto"].median()
    medianas_vpg = ventas_validas.groupby("sala")["vpg"].median()

    # Si alguna sala no tiene mediana, usar mediana global de ventas
    if mask_venta.any():
        med_global_monto = ventas_validas["monto"].median()
        med_global_vpg = ventas_validas["vpg"].median()
    else:
        med_global_monto = None
        med_global_vpg = None

    faltante_monto = mask_venta & out["monto"].isna()
    if faltante_monto.any():
        out.loc[faltante_monto, "monto"] = out.loc[faltante_monto, "sala"].map(medianas_monto)
        if out.loc[faltante_monto, "monto"].isna().any() and med_global_monto is not None and not pd.isna(med_global_monto):
            out.loc[faltante_monto & out["monto"].isna(), "monto"] = med_global_monto

    faltante_vpg = mask_venta & out["vpg"].isna()
    if faltante_vpg.any():
        out.loc[faltante_vpg, "vpg"] = out.loc[faltante_vpg, "sala"].map(medianas_vpg)
        if out.loc[faltante_vpg, "vpg"].isna().any() and med_global_vpg is not None and not pd.isna(med_global_vpg):
            out.loc[faltante_vpg & out["vpg"].isna(), "vpg"] = med_global_vpg

    return out


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def limpiar_leads(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline de limpieza de leads manteniendo la lógica de la versión base."""
    out = df.copy()

    # Parseo de fechas (mantener columnas originales como datetime cuando aplique)
    if "fecha_creacion" in out.columns:
        out["fecha_creacion"] = pd.to_datetime(out["fecha_creacion"], errors="coerce")
    if "fecha_venta" in out.columns:
        out["fecha_venta"] = pd.to_datetime(out["fecha_venta"], errors="coerce")

    out = normalizar_fuente(out)
    out = normalizar_sala(out)
    out = imputar_comentarios(out)
    out = imputar_monto_vpg(out)
    out = eliminar_duplicados(out)

    # Ordenar columnas si existen (consistencia con base)
    cols = [c for c in config.LEADS_RAW_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in cols]
    out = out[cols + extras]

    return out
