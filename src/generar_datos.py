"""Script de compatibilidad para generación de datos sintéticos."""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que src/ esté en el path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from horizonte.generar_datos import generar_y_guardar_datos  # noqa: E402
from horizonte.config import get_paths  # noqa: E402


if __name__ == "__main__":
    paths = get_paths()
    leads, fx = generar_y_guardar_datos(paths.data_dir)
    print(f"leads_ventas.csv generado: {leads}")
    print(f"fx_historico.csv generado: {fx}")
