from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from horizonte import pipeline


def _pythonpath_env():
    import os
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + prev if prev else "")
    return env


def test_pipeline_steps_y_run_all(tmp_path):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    # Generar datos
    gen = pipeline.step_generate_data(data_dir=root / "data")
    assert Path(gen["leads_path"]).exists()
    assert Path(gen["fx_path"]).exists()

    # Validar
    v_leads = pipeline.step_validate(gen["leads_path"], "leads")
    v_fx = pipeline.step_validate(gen["fx_path"], "fx")
    assert v_leads["ok"]
    assert v_fx["ok"]

    # Limpiar
    clean = pipeline.step_clean(gen["leads_path"], root / "data" / "leads_ventas_clean.csv")
    assert Path(clean["output_path"]).exists()
    assert clean["validation_ok"]
    clean_df = pd.read_csv(clean["output_path"])
    assert "fuente" in clean_df.columns

    # Forecast
    fc = pipeline.step_forecast(gen["fx_path"], root / "reports")
    assert Path(fc["metricas_path"]).exists()
    assert Path(fc["pronostico_path"]).exists()
    met = pd.read_csv(fc["metricas_path"], index_col=0)
    pron = pd.read_csv(fc["pronostico_path"])
    assert len(pron) == 6

    # Figuras
    figs = pipeline.step_report_figures(root / "reports")
    assert Path(figs["impacto"]).exists()
    assert Path(figs["roadmap"]).exists()


def test_cli_run_all(tmp_path):
    root = tmp_path / "proj_cli"
    cmd = [
        sys.executable, "-m", "horizonte.cli", "run-all",
        "--root", str(root),
        "--skip-figures",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_pythonpath_env())
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    data = root / "data"
    reports = root / "reports"
    assert (data / "leads_ventas.csv").exists()
    assert (data / "fx_historico.csv").exists()
    assert (data / "leads_ventas_clean.csv").exists()
    assert (reports / "metricas_fx.csv").exists()
    assert (reports / "pronostico_6m.csv").exists()


def test_cli_validate_falla_con_columnas_faltantes(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    cmd = [sys.executable, "-m", "horizonte.cli", "validate", "--dataset", "leads", "--input", str(bad)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_pythonpath_env())
    assert proc.returncode != 0
