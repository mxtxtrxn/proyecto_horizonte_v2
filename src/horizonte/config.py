"""Configuración y rutas por defecto del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto (carpeta que contiene src/, data/, reports/)."""
    # src/horizonte/config.py -> parents[2] == raíz del proyecto
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    reports_dir: Path
    notebooks_dir: Path

    @property
    def leads_raw(self) -> Path:
        return self.data_dir / "leads_ventas.csv"

    @property
    def leads_clean(self) -> Path:
        return self.data_dir / "leads_ventas_clean.csv"

    @property
    def fx_historico(self) -> Path:
        return self.data_dir / "fx_historico.csv"

    @property
    def metricas_fx(self) -> Path:
        return self.reports_dir / "metricas_fx.csv"

    @property
    def pronostico_6m(self) -> Path:
        return self.reports_dir / "pronostico_6m.csv"


def get_paths(root: Path | None = None) -> Paths:
    base = root or get_project_root()
    return Paths(
        root=base,
        data_dir=base / "data",
        reports_dir=base / "reports",
        notebooks_dir=base / "notebooks",
    )


# Parámetros por defecto de generación
LEADS_N = 5000
LEADS_DUP_RATIO = 0.05
LEADS_EXTRA_NULL_RATIO = 0.10
FX_START = "2014-01-01"
FX_END = "2024-12-01"
FX_NULL_RATIO = 0.02

# Semillas para reproducibilidad
SEED_RANDOM = 42
SEED_NUMPY = 42
SEED_FX = 7

# Esquemas esperados
LEADS_RAW_COLUMNS = [
    "lead_id",
    "fuente",
    "fecha_creacion",
    "fecha_venta",
    "monto",
    "vpg",
    "sala",
    "comentarios",
]

LEADS_CLEAN_COLUMNS = LEADS_RAW_COLUMNS.copy()

FX_COLUMNS = ["fecha", "tipo_cambio", "tasa_interes", "inflacion"]

# Categorías canónicas (alineadas con la versión base)
FUENTE_MAP = {
    "proveedor a": "Proveedor A",
    "proveedora": "Proveedor A",
    "portal web": "Portal Web",
    "social": "Social",
    "facebook": "Social",
    "fb": "Social",
    "calle": "Calle",
    "calle tms": "Calle",
    "exchange": "Exchange",
    "ota": "OTA",
    "hotel": "Hotel",
}

SALA_MAP = {
    "cancun": "Cancún",
    "playa del carmen": "Playa del Carmen",
    "pdc": "Playa del Carmen",
    "puerto vallarta": "Puerto Vallarta",
    "pv": "Puerto Vallarta",
    "los cabos": "Los Cabos",
    "cabo": "Los Cabos",
}

FUENTE_CANONICAS = ["Proveedor A", "Portal Web", "Social", "Calle", "Exchange", "OTA", "Hotel", "Desconocido"]
SALA_CANONICAS = ["Cancún", "Playa del Carmen", "Puerto Vallarta", "Los Cabos", "Desconocido"]
