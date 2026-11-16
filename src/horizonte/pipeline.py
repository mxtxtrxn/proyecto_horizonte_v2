"""Orquestación del pipeline de datos, limpieza y forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .cleaning import limpiar_leads
from .forecasting import ejecutar_forecasting
from .generar_datos import generar_y_guardar_datos
from .io_utils import read_csv, write_csv
from .reporting import generar_figura_impacto, generar_figura_roadmap, guardar_metricas, guardar_pronostico
from .validation import validate


def step_generate_data(data_dir: Path | None = None) -> dict[str, Any]:
    leads_path, fx_path = generar_y_guardar_datos(data_dir=data_dir)
    return {"leads_path": str(leads_path), "fx_path": str(fx_path)}


def step_validate(input_path: Path | str, dataset: str) -> dict[str, Any]:
    df = read_csv(input_path)
    result = validate(dataset, df)
    return {
        "dataset": result.dataset,
        "ok": result.ok,
        "errors": len(result.errors),
        "warnings": len(result.warnings),
        "summary": result.summary(),
    }


def step_clean(input_path: Path | str, output_path: Path | str) -> dict[str, Any]:
    df = read_csv(input_path)
    df_clean = limpiar_leads(df)
    out = write_csv(df_clean, output_path, index=False)
    result = validate("leads_clean", df_clean)
    return {
        "output_path": str(out),
        "rows_in": int(len(df)),
        "rows_out": int(len(df_clean)),
        "validation_ok": result.ok,
        "validation_errors": len(result.errors),
        "validation_warnings": len(result.warnings),
    }


def step_forecast(input_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    df_fx = read_csv(input_path)
    # Validación básica previa
    val = validate("fx", df_fx)
    if not val.ok:
        raise ValueError(f"Validación FX fallida:\n{val.summary()}")

    res = ejecutar_forecasting(df_fx)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metricas_path = guardar_metricas(res.metricas, output_dir / "metricas_fx.csv")
    pron_path = guardar_pronostico(res.pronostico, output_dir / "pronostico_6m.csv")
    return {
        "metricas_path": str(metricas_path),
        "pronostico_path": str(pron_path),
        "split": res.split_info,
    }


def step_report_figures(output_dir: Path | str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    impacto = generar_figura_impacto(output_dir / "diapositiva4_impacto.png")
    roadmap = generar_figura_roadmap(output_dir / "diapositiva5_roadmap.png")
    return {"impacto": str(impacto), "roadmap": str(roadmap)}


def run_all(root: Path | None = None, generate_figures: bool = True) -> dict[str, Any]:
    paths = config.get_paths(root)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    results["generate_data"] = step_generate_data(paths.data_dir)

    # Validaciones de insumos
    results["validate_leads_raw"] = step_validate(paths.leads_raw, "leads")
    results["validate_fx"] = step_validate(paths.fx_historico, "fx")

    # Limpieza
    results["clean"] = step_clean(paths.leads_raw, paths.leads_clean)

    # Forecast
    results["forecast"] = step_forecast(paths.fx_historico, paths.reports_dir)

    # Figuras (regenerables)
    if generate_figures:
        results["figures"] = step_report_figures(paths.reports_dir)

    return results
