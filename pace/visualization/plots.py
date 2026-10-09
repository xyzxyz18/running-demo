"""Static dashboard/report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def _configure_font() -> None:
    """Prefer an installed CJK font so Chinese feedback renders in the report."""
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "PingFang HK", "Hiragino Sans GB", "Microsoft YaHei",
                 "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def create_report(path: Path, times: np.ndarray, angles: Dict[str, np.ndarray],
                  foot_y: Dict[str, np.ndarray], events: Dict[str, List[int]],
                  metrics: Dict[str, object], feedback: List[str]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    _configure_font()
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.5, 1))
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[:, 1])
    ax1.plot(times, angles["left_knee"], label="Left knee", lw=1.7)
    ax1.plot(times, angles["right_knee"], label="Right knee", lw=1.7)
    ax1.set(title="Knee angle", xlabel="Time (s)", ylabel="Angle (deg)")
    ax1.legend()
    ax2.plot(times, foot_y["left"], label="Left ankle")
    ax2.plot(times, foot_y["right"], label="Right ankle")
    for side, color in (("left", "C0"), ("right", "C1")):
        strike_ids = events[f"{side}_strikes"]
        if strike_ids:
            ax2.scatter(times[strike_ids], foot_y[side][strike_ids], marker="v",
                        color=color, edgecolor="black", zorder=5, label=f"{side.title()} strike")
    ax2.set(title="Ankle clearance and strikes", xlabel="Time (s)", ylabel="Clearance / leg length")
    ax2.legend(ncol=2, fontsize=8)
    ax3.axis("off")
    ax3.set_title("Running analysis dashboard", fontsize=18, fontweight="bold", loc="left")
    labels = [
        ("Cadence", metrics.get("cadence_steps_per_min"), "steps/min"),
        ("Step time", metrics.get("step_time_seconds"), "s"),
        ("Stride time", metrics.get("stride_time_seconds"), "s"),
        ("Knee ROM", metrics.get("knee_rom_degrees"), "deg"),
        ("Hip ROM", metrics.get("hip_rom_degrees"), "deg"),
        ("Stride asymmetry", metrics.get("stride_time_asymmetry_percent"), "%"),
    ]
    y = 0.88
    for label, value, unit in labels:
        shown = "--" if value is None else str(value)
        ax3.text(0.02, y, label, fontsize=11, color="#607080", transform=ax3.transAxes)
        ax3.text(0.98, y, f"{shown} {unit}", fontsize=15, ha="right",
                 fontweight="bold", transform=ax3.transAxes)
        y -= 0.09
    ax3.text(0.02, y - 0.02, "Feedback", fontsize=14, fontweight="bold", transform=ax3.transAxes)
    y -= 0.085
    for item in feedback:
        # English heading/layout remains portable; Chinese feedback is also in JSON.
        ax3.text(0.04, y, "• " + item, fontsize=10, va="top", wrap=True,
                 transform=ax3.transAxes)
        y -= 0.09
    fig.suptitle("Running Pose Analysis", fontsize=20, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)
