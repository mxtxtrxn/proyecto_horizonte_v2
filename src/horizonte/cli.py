"""Interfaz de línea de comandos del proyecto."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

# Permitir ejecución directa desde la raíz del proyecto sin instalación
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from horizonte import config  # noqa: E402
from horizonte import pipeline  # noqa: E402

app = typer.Typer(help="CLI de Proyecto Horizonte v2 (datos, validación, limpieza y forecasting)")


@app.command("generate-data")
def generate_data(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Directorio de salida para datos CSV"),
):
    """Genera datos sintéticos de leads y FX."""
    try:
        res = pipeline.step_generate_data(data_dir=data_dir)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        typer.secho(f"Error al generar datos: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Tipo de dataset: leads|leads_clean|fx"),
    input_path: Path = typer.Option(..., "--input", "-i", help="Ruta al archivo CSV a validar"),
):
    """Valida esquema y calidad de un dataset."""
    try:
        res = pipeline.step_validate(input_path=input_path, dataset=dataset)
        typer.echo(res["summary"])
        if not res["ok"]:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error en validación: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("clean")
def clean_cmd(
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="CSV de leads crudo"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="CSV de salida limpio"),
):
    """Limpia y normaliza el dataset de leads."""
    paths = config.get_paths()
    in_path = input_path or paths.leads_raw
    out_path = output_path or paths.leads_clean
    try:
        res = pipeline.step_clean(input_path=in_path, output_path=out_path)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        if not res.get("validation_ok", False):
            typer.secho("La validación del dataset limpio reportó errores.", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error en limpieza: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("forecast")
def forecast_cmd(
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="CSV histórico FX"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directorio para métricas y pronóstico"),
):
    """Ejecuta el pipeline de forecasting FX."""
    paths = config.get_paths()
    in_path = input_path or paths.fx_historico
    out_dir = output_dir or paths.reports_dir
    try:
        res = pipeline.step_forecast(input_path=in_path, output_dir=out_dir)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        typer.secho(f"Error en forecasting: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("report-figures")
def report_figures_cmd(
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directorio de salida para figuras"),
):
    """Regenera figuras de reporte."""
    paths = config.get_paths()
    out_dir = output_dir or paths.reports_dir
    try:
        res = pipeline.step_report_figures(output_dir=out_dir)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        typer.secho(f"Error generando figuras: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("run-all")
def run_all_cmd(
    root: Optional[Path] = typer.Option(None, "--root", help="Raíz del proyecto"),
    skip_figures: bool = typer.Option(False, "--skip-figures", help="Omitir generación de figuras"),
):
    """Ejecuta el pipeline completo (generar, validar, limpiar, forecast y figuras)."""
    try:
        res = pipeline.run_all(root=root, generate_figures=not skip_figures)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        typer.secho(f"Error en run-all: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
