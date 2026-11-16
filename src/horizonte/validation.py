"""Validación de esquema y calidad de datos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config
from .io_utils import ensure_columns


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    dataset: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add(self, level: str, code: str, message: str, **details: Any) -> None:
        self.issues.append(ValidationIssue(level=level, code=code, message=message, details=details))

    def summary(self) -> str:
        lines = [f"Dataset: {self.dataset}", f"Errores: {len(self.errors)} | Advertencias: {len(self.warnings)}"]
        for i in self.issues:
            prefix = "ERROR" if i.level == "error" else "WARN"
            lines.append(f"[{prefix}] {i.code}: {i.message}")
        return "\n".join(lines)


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def validate_leads_raw(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(dataset="leads_raw")
    missing, extra = ensure_columns(df, config.LEADS_RAW_COLUMNS)
    if missing:
        res.add("error", "missing_columns", f"Faltan columnas requeridas: {missing}", missing=missing)
    if extra:
        res.add("warning", "extra_columns", f"Columnas adicionales no esperadas: {extra}", extra=extra)

    if df.empty:
        res.add("error", "empty_dataset", "El dataset está vacío")
        return res

    # lead_id
    if "lead_id" in df.columns:
        if df["lead_id"].isna().any():
            res.add("error", "lead_id_null", "Existen lead_id nulos")
        if df["lead_id"].duplicated().any():
            dup = int(df["lead_id"].duplicated().sum())
            res.add("warning", "lead_id_duplicates", f"Existen {dup} lead_id duplicados (pueden ser registros repetidos)", count=dup)

    # Fechas
    for col in ["fecha_creacion", "fecha_venta"]:
        if col in df.columns:
            parsed = _to_datetime_safe(df[col])
            # fecha_venta puede ser nula; fecha_creacion debería ser válida
            invalid = parsed.notna().sum() == 0 and df[col].notna().any()
            if col == "fecha_creacion":
                bad = df[col].notna() & parsed.isna()
                if bad.any():
                    res.add("error", "invalid_dates", f"Valores no parseables en {col}", count=int(bad.sum()))
            else:
                bad = df[col].notna() & parsed.isna()
                if bad.any():
                    res.add("warning", "invalid_dates", f"Valores no parseables en {col}", count=int(bad.sum()))

    # Montos/VPG: si hay valores, deben ser numéricos y no negativos cuando presentes
    for col in ["monto", "vpg"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_numeric = df[col].notna() & numeric.isna()
            if non_numeric.any():
                res.add("error", "non_numeric", f"Valores no numéricos en {col}", count=int(non_numeric.sum()))
            negative = numeric < 0
            if negative.fillna(False).any():
                res.add("error", "negative_values", f"Valores negativos en {col}", count=int(negative.fillna(False).sum()))

    # Coherencia básica: venta implica posibilidad de monto/vpg (pueden faltar y se imputan en clean)
    if {"fecha_venta", "monto", "vpg"}.issubset(df.columns):
        sin_venta_con_monto = df["fecha_venta"].isna() & df["monto"].notna()
        if sin_venta_con_monto.any():
            res.add("warning", "monto_without_sale", "Registros sin fecha_venta pero con monto", count=int(sin_venta_con_monto.sum()))

    # Duplicados exactos
    exact_dups = int(df.duplicated().sum())
    if exact_dups > 0:
        res.add("warning", "exact_duplicates", f"Se detectaron {exact_dups} filas duplicadas exactas", count=exact_dups)

    return res


def validate_leads_clean(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(dataset="leads_clean")
    missing, extra = ensure_columns(df, config.LEADS_RAW_COLUMNS)
    if missing:
        res.add("error", "missing_columns", f"Faltan columnas requeridas: {missing}", missing=missing)
    if extra:
        res.add("warning", "extra_columns", f"Columnas adicionales: {extra}", extra=extra)

    if df.empty:
        res.add("error", "empty_dataset", "El dataset limpio está vacío")
        return res

    if "lead_id" in df.columns and not df["lead_id"].is_unique:
        res.add("error", "lead_id_not_unique", "lead_id debe ser único tras limpieza")

    # Sin duplicados exactos
    exact_dups = int(df.duplicated().sum())
    if exact_dups > 0:
        res.add("error", "exact_duplicates", "No deben existir duplicados exactos en el dataset limpio", count=exact_dups)

    # Categóricas canónicas
    if "fuente" in df.columns:
        invalid_fuente = set(df["fuente"].dropna().unique()) - set(config.FUENTE_CANONICAS)
        if invalid_fuente:
            res.add("error", "invalid_fuente", f"Valores de fuente fuera de catálogo: {sorted(invalid_fuente)}")
    if "sala" in df.columns:
        invalid_sala = set(df["sala"].dropna().unique()) - set(config.SALA_CANONICAS)
        if invalid_sala:
            res.add("error", "invalid_sala", f"Valores de sala fuera de catálogo: {sorted(invalid_sala)}")

    # Comentarios sin nulos
    if "comentarios" in df.columns and df["comentarios"].isna().any():
        res.add("error", "comentarios_null", "No deben quedar comentarios nulos tras limpieza")

    # Coherencia venta/monto/vpg: si no hay venta, monto y vpg deben ser nulos
    if {"fecha_venta", "monto", "vpg"}.issubset(df.columns):
        sin_venta = df["fecha_venta"].isna()
        con_monto_sin_venta = sin_venta & df["monto"].notna()
        con_vpg_sin_venta = sin_venta & df["vpg"].notna()
        if con_monto_sin_venta.any():
            res.add("error", "monto_without_sale", "Registros sin venta con monto presente", count=int(con_monto_sin_venta.sum()))
        if con_vpg_sin_venta.any():
            res.add("error", "vpg_without_sale", "Registros sin venta con vpg presente", count=int(con_vpg_sin_venta.sum()))

        # Montos/VPG positivos cuando existen
        for col in ["monto", "vpg"]:
            vals = pd.to_numeric(df[col], errors="coerce")
            nonpos = vals.notna() & (vals <= 0)
            if nonpos.any():
                res.add("error", "non_positive_values", f"Valores <= 0 en {col}", count=int(nonpos.sum()))

    return res


def validate_fx(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(dataset="fx_historico")
    missing, extra = ensure_columns(df, config.FX_COLUMNS)
    if missing:
        res.add("error", "missing_columns", f"Faltan columnas requeridas: {missing}", missing=missing)
    if extra:
        res.add("warning", "extra_columns", f"Columnas adicionales: {extra}", extra=extra)

    if df.empty:
        res.add("error", "empty_dataset", "El dataset FX está vacío")
        return res

    if "fecha" in df.columns:
        fechas = _to_datetime_safe(df["fecha"])
        bad = df["fecha"].notna() & fechas.isna()
        if bad.any():
            res.add("error", "invalid_dates", "Fechas no parseables en columna fecha", count=int(bad.sum()))
        if fechas.duplicated().any():
            res.add("warning", "duplicate_dates", "Existen fechas duplicadas", count=int(fechas.duplicated().sum()))

    for col in ["tipo_cambio", "tasa_interes", "inflacion"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_numeric = df[col].notna() & numeric.isna()
            if non_numeric.any():
                res.add("error", "non_numeric", f"Valores no numéricos en {col}", count=int(non_numeric.sum()))
            if col == "tipo_cambio" and df[col].isna().any():
                res.add("warning", "nulls", "Existen nulos en tipo_cambio (se pueden interpolar en el pipeline)", count=int(df[col].isna().sum()))

    # Rango temporal esperado (consistente con base)
    if "fecha" in df.columns:
        fechas = _to_datetime_safe(df["fecha"]).dropna()
        if not fechas.empty:
            if fechas.min() > pd.Timestamp("2014-01-31") or fechas.max() < pd.Timestamp("2024-11-30"):
                res.add("warning", "date_range", "El rango de fechas difiere del esperado (2014-01 a 2024-12)", min=str(fechas.min().date()), max=str(fechas.max().date()))

    return res


def validate(dataset: str, df: pd.DataFrame) -> ValidationResult:
    dataset = dataset.lower().strip()
    if dataset in {"leads", "leads_raw", "raw"}:
        return validate_leads_raw(df)
    if dataset in {"leads_clean", "clean"}:
        return validate_leads_clean(df)
    if dataset in {"fx", "fx_historico"}:
        return validate_fx(df)
    raise ValueError(f"Dataset de validación no soportado: {dataset}")
