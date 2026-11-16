"""Generación de artefactos de reporte (CSV y figuras)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from .io_utils import ensure_parent_dir, write_csv


def guardar_metricas(metricas: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    ensure_parent_dir(p)
    metricas.to_csv(p)
    return p


def guardar_pronostico(pronostico: pd.DataFrame, path: Path | str) -> Path:
    return write_csv(pronostico, path, index=False)


def generar_figura_impacto(output_path: Path | str) -> Path:
    p = Path(output_path)
    ensure_parent_dir(p)

    canales = [
        "Hotel", "Social", "Exchange", "Calle",
        "OTA", "Desconocido", "Portal Web", "Proveedor A",
    ]
    tasas = [62.6, 61.6, 60.9, 60.5, 59.7, 59.1, 58.6, 57.5]
    kpis = [
        ("$132,614 MXN", "promedio por venta"),
        ("45 dias", "promedio de conversion"),
        ("59.9%", "tasa de conversion global"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })

    color_primary = "#1F4E79"
    color_secondary = "#5B9BD5"
    color_light = "#B8CCE4"
    color_grid = "#E8E8E8"
    color_text = "#333333"
    color_accent = "#ED7D31"
    colores = [color_primary] * 3 + [color_secondary] * 2 + [color_light] * 3

    fig = plt.figure(figsize=(14, 8), dpi=100)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1.2], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    y_pos = np.arange(len(canales))
    bars = ax.barh(y_pos, tasas, color=colores, edgecolor="white", height=0.7)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(canales, fontsize=13, color=color_text)
    ax.set_xlim(55, 65)
    ax.set_xticks([55, 57, 59, 61, 63, 65])
    ax.set_xticklabels([f"{v}%" for v in [55, 57, 59, 61, 63, 65]], fontsize=11, color=color_text)
    ax.xaxis.grid(True, color=color_grid, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)

    for bar, tasa in zip(bars, tasas):
        ax.text(tasa + 0.15, bar.get_y() + bar.get_height() / 2, f"{tasa}%",
                va="center", ha="left", fontsize=12, fontweight="bold", color=color_text)

    ax.axvline(59.9, color=color_accent, linestyle="--", linewidth=2, alpha=0.8)
    ax.text(59.9, -0.9, "   Promedio global: 59.9%", color=color_accent,
            fontsize=11, fontweight="bold", ha="left", va="center")
    ax.set_title("Tasa de conversion por fuente", fontsize=16, fontweight="bold",
                 color=color_text, loc="left", pad=15)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    ax_kpi = fig.add_subplot(gs[1])
    ax_kpi.axis("off")
    n_kpis = len(kpis)
    col_width = 1 / n_kpis
    for i, (valor, etiqueta) in enumerate(kpis):
        x_center = (i + 0.5) * col_width
        box = FancyBboxPatch((i * col_width + 0.02, 0.15), col_width - 0.04, 0.7,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=0, facecolor="#F2F2F2", transform=ax_kpi.transAxes)
        ax_kpi.add_patch(box)
        ax_kpi.text(x_center, 0.60, valor, ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color_primary, transform=ax_kpi.transAxes)
        ax_kpi.text(x_center, 0.32, etiqueta, ha="center", va="center",
                    fontsize=12, color=color_text, transform=ax_kpi.transAxes)

    fig.suptitle("Lo que encontramos en los datos de Horizonte", fontsize=20,
                 fontweight="bold", color=color_text, x=0.06, y=0.98, ha="left")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return p


def generar_figura_roadmap(output_path: Path | str) -> Path:
    p = Path(output_path)
    ensure_parent_dir(p)

    fases = [
        {"periodo": "Mes 1-2", "titulo": "Datos en produccion", "detalle": "+ pipeline ETL automatizado", "color": "#1F4E79"},
        {"periodo": "Mes 3-4", "titulo": "Modelo FX + verificador", "detalle": "backtest + supervision", "color": "#5B9BD5"},
        {"periodo": "Mes 5-6", "titulo": "Agente call center live", "detalle": "con CRM principal + TMS", "color": "#ED7D31"},
    ]
    riesgos = [
        {"color": "#FFC000", "titulo": "Adopcion del equipo", "descripcion": "Capacitacion de 1 semana\npor area antes de cada fase"},
        {"color": "#FFC000", "titulo": "Calidad de datos futura", "descripcion": "Reglas de captura\n+ validacion mensual automatica"},
        {"color": "#FFC000", "titulo": "Cambios en el negocio", "descripcion": "Modelos se actualizan\ntrimestralmente con datos frescos"},
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    color_text = "#333333"
    color_text_light = "#666666"
    color_line = "#CCCCCC"
    color_bg_card = "#F8F8F8"

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.15, top=0.88, bottom=0.05, left=0.05, right=0.95)

    ax_tl = fig.add_subplot(gs[0])
    ax_tl.set_xlim(0, 1); ax_tl.set_ylim(0, 1); ax_tl.axis("off")
    y_line = 0.65
    ax_tl.plot([0.10, 0.90], [y_line, y_line], color=color_line, linewidth=3, zorder=1, transform=ax_tl.transAxes)
    x_positions = [0.20, 0.50, 0.80]
    card_width = 0.24
    card_height = 0.42

    for i, (fase, x) in enumerate(zip(fases, x_positions)):
        circle = Circle((x, y_line), radius=0.028, facecolor=fase["color"], edgecolor="white", linewidth=2.5, zorder=3, transform=ax_tl.transAxes)
        ax_tl.add_patch(circle)
        ax_tl.text(x, y_line, str(i + 1), ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=4, transform=ax_tl.transAxes)
        ax_tl.text(x, y_line + 0.05, fase["periodo"], ha="center", va="bottom", fontsize=13, fontweight="bold", color=fase["color"], transform=ax_tl.transAxes)
        card_y = y_line - 0.06 - card_height
        card_x = x - card_width / 2
        box = FancyBboxPatch((card_x, card_y), card_width, card_height, boxstyle="round,pad=0.005,rounding_size=0.02",
                             linewidth=2, facecolor="white", edgecolor=fase["color"], zorder=2, transform=ax_tl.transAxes)
        ax_tl.add_patch(box)
        ax_tl.text(x, card_y + card_height * 0.75, fase["titulo"], ha="center", va="center", fontsize=10.5, fontweight="bold", color=color_text, transform=ax_tl.transAxes)
        ax_tl.text(x, card_y + card_height * 0.28, fase["detalle"], ha="center", va="center", fontsize=9, color=color_text_light, transform=ax_tl.transAxes)

    ax_rg = fig.add_subplot(gs[1])
    ax_rg.set_xlim(0, 1); ax_rg.set_ylim(0, 1); ax_rg.axis("off")
    card_width = 0.30
    card_height = 0.85
    gap = 0.015
    total_width = card_width * 3 + gap * 2
    start_x = (1 - total_width) / 2
    card_y = 0.08

    for i, riesgo in enumerate(riesgos):
        card_x = start_x + i * (card_width + gap)
        box = FancyBboxPatch((card_x, card_y), card_width, card_height, boxstyle="round,pad=0.005,rounding_size=0.015",
                             linewidth=0, facecolor=color_bg_card, zorder=1, transform=ax_rg.transAxes)
        ax_rg.add_patch(box)
        barra = Rectangle((card_x, card_y), 0.008, card_height, linewidth=0, facecolor=riesgo["color"], zorder=2, transform=ax_rg.transAxes)
        ax_rg.add_patch(barra)
        icon_x = card_x + 0.045
        icon_y = card_y + card_height - 0.10
        circle = Circle((icon_x, icon_y), radius=0.022, facecolor=riesgo["color"], edgecolor="white", linewidth=1.5, zorder=3, transform=ax_rg.transAxes)
        ax_rg.add_patch(circle)
        ax_rg.text(icon_x, icon_y, "!", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=4, transform=ax_rg.transAxes)
        ax_rg.text(card_x + 0.075, icon_y, riesgo["titulo"], ha="left", va="center", fontsize=10.5, fontweight="bold", color=color_text, zorder=4, transform=ax_rg.transAxes)
        sep_y = icon_y - 0.075
        ax_rg.plot([card_x + 0.03, card_x + card_width - 0.02], [sep_y, sep_y], color="#E0E0E0", linewidth=1, zorder=3, transform=ax_rg.transAxes)
        ax_rg.text(card_x + 0.03, sep_y - 0.04, riesgo["descripcion"], ha="left", va="top", fontsize=9, color=color_text_light, linespacing=1.5, zorder=4, transform=ax_rg.transAxes)

    fig.suptitle("Que sigue y que debemos cuidar?", fontsize=22, fontweight="bold", color=color_text, x=0.05, y=0.96, ha="left")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return p
