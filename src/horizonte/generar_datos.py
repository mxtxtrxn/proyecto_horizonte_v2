"""Generación de datos sintéticos para leads y serie FX."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io_utils import write_csv


def generar_leads(
    n: int = config.LEADS_N,
    seed_random: int = config.SEED_RANDOM,
    dup_ratio: float = config.LEADS_DUP_RATIO,
    extra_null_ratio: float = config.LEADS_EXTRA_NULL_RATIO,
) -> pd.DataFrame:
    random.seed(seed_random)

    fuentes_raw = [
        "Proveedor A", "proveedor a", "PROVEEDOR A", "ProveedorA",
        "Portal Web", "portal web", "PORTAL WEB",
        "Social", "social", "Facebook", "FB",
        "Calle", "calle", "Calle TMS",
        "Exchange", "exchange", "OTA", "ota",
        "Hotel", "hotel", None, "", "N/A",
    ]

    salas = [
        "Cancún", "cancún", "CANCÚN", "Playa del Carmen", "PDC",
        "Puerto Vallarta", "PV", "Los Cabos", "Cabo", "cabo",
    ]

    comentarios_ok = [
        "Cliente interesado, pidió información por email.",
        "No contesta llamadas, dejar seguimiento.",
        "Solicitó cotización para 2 adultos y 2 niños.",
        "Muy interesado, agendar presentación.",
        "Pidió hablar con supervisor.",
        "Ya había hecho tour antes.",
        "Prefiere destino de playa.",
        "Objeción de precio, volver a llamar.",
        "Cliente molesto por llamadas repetidas.",
        "Solicitó cancelar su registro.",
    ]

    data = []
    for i in range(n):
        lead_id = 100000 + i
        fuente = random.choice(fuentes_raw)
        fecha_creacion = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))

        if random.random() < 0.60:
            fecha_venta = fecha_creacion + timedelta(days=random.randint(1, 90))
            monto = round(random.uniform(15000, 250000), 2)
            vpg = round(random.uniform(500, 8000), 2)
        else:
            fecha_venta = None
            monto = None
            vpg = None

        sala = random.choice(salas)
        comentario = random.choice(comentarios_ok) if random.random() < 0.7 else None

        data.append([lead_id, fuente, fecha_creacion, fecha_venta, monto, vpg, sala, comentario])

    # Duplicados intencionales
    if dup_ratio > 0:
        duplicados = random.sample(data, int(n * dup_ratio))
        data.extend(duplicados)

    # Nulos adicionales en monto/vpg
    for row in data:
        if random.random() < extra_null_ratio and row[4] is not None:
            row[4] = None
        if random.random() < extra_null_ratio and row[5] is not None:
            row[5] = None

    df = pd.DataFrame(data, columns=config.LEADS_RAW_COLUMNS)
    return df


def generar_fx(
    start: str = config.FX_START,
    end: str = config.FX_END,
    seed_fx: int = config.SEED_FX,
    null_ratio: float = config.FX_NULL_RATIO,
) -> pd.DataFrame:
    fechas = pd.date_range(start=start, end=end, freq="MS")
    n_fx = len(fechas)

    np.random.seed(seed_fx)
    tendencia = np.linspace(13.0, 20.0, n_fx)
    estacionalidad = 0.5 * np.sin(np.linspace(0, 8 * np.pi, n_fx))
    ruido = np.random.normal(0, 0.4, n_fx)
    tipo_cambio = tendencia + estacionalidad + ruido

    tasa_interes = 4.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, n_fx)) + np.random.normal(0, 0.3, n_fx)
    inflacion = 3.5 + 1.5 * np.sin(np.linspace(0, 6 * np.pi, n_fx)) + np.random.normal(0, 0.2, n_fx)

    df = pd.DataFrame(
        {
            "fecha": fechas.strftime("%Y-%m-%d"),
            "tipo_cambio": np.round(tipo_cambio, 4),
            "tasa_interes": np.round(tasa_interes, 4),
            "inflacion": np.round(inflacion, 4),
        }
    )

    if null_ratio > 0:
        idx_nulos = np.random.choice(df.index, size=max(1, int(n_fx * null_ratio)), replace=False)
        df.loc[idx_nulos, "tipo_cambio"] = None

    return df


def generar_y_guardar_datos(
    data_dir: Path | None = None,
    n_leads: int = config.LEADS_N,
    seed_random: int = config.SEED_RANDOM,
    seed_fx: int = config.SEED_FX,
) -> tuple[Path, Path]:
    paths = config.get_paths()
    target_dir = data_dir or paths.data_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    df_leads = generar_leads(n=n_leads, seed_random=seed_random)
    df_fx = generar_fx(seed_fx=seed_fx)

    ruta_leads = write_csv(df_leads, target_dir / "leads_ventas.csv")
    ruta_fx = write_csv(df_fx, target_dir / "fx_historico.csv")
    return ruta_leads, ruta_fx


if __name__ == "__main__":
    paths = config.get_paths()
    l, f = generar_y_guardar_datos(paths.data_dir)
    print(f"leads_ventas.csv generado: {l}")
    print(f"fx_historico.csv generado: {f}")
