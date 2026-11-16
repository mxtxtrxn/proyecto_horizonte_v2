# Proyecto Horizonte v2: Análisis de Datos y Forecasting

Versión ampliada y modular del proyecto Horizonte. Incluye generación de datos
sintéticos, validación, limpieza de leads, forecasting de tipo de cambio
MXN/USD, interfaz de línea de comandos (CLI) y suite de pruebas automatizadas.

**Datos:** sintéticos. **Contexto:** caso de estudio con nombres y escenarios ficticios.

## Estructura del repositorio

```text
proyecto_horitonte_v2/
├─ data/                    # Datos de entrada y salida (CSV)
├─ notebooks/               # Notebooks de análisis (opcionales)
├─ reports/                 # Métricas, pronósticos y figuras generadas
├─ src/
│  ├─ horizonte/            # Paquete principal (módulos refactorizados)
│  └─ generar_datos.py      # Script de compatibilidad para generar datos
├─ tests/                   # Pruebas automatizadas (pytest)
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## Requisitos previos

- Python 3.11+
- pip

## Instalación

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso (CLI)

Ejecutar desde la raíz del proyecto. Si el paquete no está instalado, asegúrate de incluir `src` en `PYTHONPATH`.

**Linux/macOS:**
```bash
export PYTHONPATH=src
python -m horizonte.cli generate-data
python -m horizonte.cli validate --dataset leads --input data/leads_ventas.csv
python -m horizonte.cli clean --input data/leads_ventas.csv --output data/leads_ventas_clean.csv
python -m horizonte.cli forecast --input data/fx_historico.csv --output-dir reports
python -m horizonte.cli report-figures --output-dir reports
python -m horizonte.cli run-all
# Omitir figuras: python -m horizonte.cli run-all --skip-figures
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "src"
python -m horizonte.cli generate-data
python -m horizonte.cli validate --dataset leads --input data/leads_ventas.csv
python -m horizonte.cli clean --input data/leads_ventas.csv --output data/leads_ventas_clean.csv
python -m horizonte.cli forecast --input data/fx_historico.csv --output-dir reports
python -m horizonte.cli report-figures --output-dir reports
python -m horizonte.cli run-all
# Omitir figuras: python -m horizonte.cli run-all --skip-figures
```

También se puede usar el script de compatibilidad:

```bash
python src/generar_datos.py
```

## Pruebas

```bash
pytest -q
```

## Salidas principales

- `data/leads_ventas.csv`: dataset de leads sintético (entrada)
- `data/fx_historico.csv`: serie histórica MXN/USD sintética (entrada)
- `data/leads_ventas_clean.csv`: leads limpios y normalizados
- `reports/metricas_fx.csv`: métricas de modelos de forecasting
- `reports/pronostico_6m.csv`: pronóstico a 6 meses

## Notas

- Los datos son sintéticos y se generan con semillas para favorecer reproducibilidad.
- La validación reporta errores y advertencias de esquema/calidad según el dataset.
- Este repositorio no incluye carpeta `.git` en los artefactos de entrega.
