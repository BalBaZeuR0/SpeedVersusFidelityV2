# SPDX-License-Identifier: BSD-3-Clause
"""Draw the method overview figure of the paper: the lander model, the shared loop with its two plants, and the checks of one episode.

Usage::

    python scripts/draw_method_figure.py

Writes ``plots/method_overview.png`` and ``plots/method_overview.pdf``. Every number on the figure
comes from the LunarSim v2 Apollo LM preset (``configs/landers/apollo_lunar_module.yaml``), the
touchdown metric (``configs/metrics/touchdown.yaml``), the S1 start box
(``configs/start/terminal_box.yaml``) and the scenario clock (200 Hz physics, 50 Hz control).
Needs only matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
INK, SECONDARY, MUTED, LINE = "#1f1f1f", "#52514e", "#898781", "#6d6c68"
PANEL_FILL, PANEL_EDGE, BODY_FILL = "#f4f4f2", "#bdbcb5", "#e2e2df"
PHYSX, ANALYTIC = "#2a78d6", "#eb6834"  # simulator colours of every result figure
CHECK_FILL, CHECK_EDGE = "#e4f2e7", "#4c9a62"
FLIGHT_FILL, FLIGHT_EDGE = "#e3edf8", "#5b8fcf"
NOTE = "#c0392b"

# Apollo LM preset (metres, footpad plane at z = 0).
FOOTPAD_RADIUS, FOOTPAD_DIAMETER, FOOTPAD_DEPTH = 4.256, 0.940, 0.178
DESCENT_BOTTOM, DESCENT_HEIGHT, DESCENT_WIDTH = 1.430, 1.749, 4.216
ASCENT_DEPTH, ASCENT_HEIGHT = 4.039, 2.832
NOZZLE_EXIT_DIAMETER, NOZZLE_EXIT_HEIGHT = 1.499, 0.483
CENTRE_OF_MASS_HEIGHT = 3.62
RCS_CLUSTER_ARM, RCS_CLUSTER_HEIGHT = 1.679, CENTRE_OF_MASS_HEIGHT + 1.029


def panel(figure, rectangle, title: str):
    """Add a rounded grey panel with a bold title and return an axes inside it (axis off)."""
    left, bottom, width, height = rectangle
    figure.patches.append(
        FancyBboxPatch((left, bottom), width, height, boxstyle="round,pad=0,rounding_size=0.012",
                       transform=figure.transFigure, facecolor=PANEL_FILL, edgecolor=PANEL_EDGE, linewidth=1.0, zorder=-10)
    )
    figure.text(left + 0.012, bottom + height - 0.03, title, fontsize=12, fontweight="bold", color=SECONDARY, va="top")
    axes = figure.add_axes((left + 0.01, bottom + 0.01, width - 0.02, height - 0.06), zorder=5)
    axes.set_axis_off()
    return axes


def label(axes, text: str, point, text_position, align="left", color=INK) -> None:
    """Write ``text`` at ``text_position`` with a thin leader line to ``point``."""
    axes.annotate(text, point, xytext=text_position, fontsize=9.5, color=color, ha=align, va="center",
                  arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=1))


def box(axes, centre, size, text: str, fill: str, edge: str, fontsize=9.5) -> None:
    """Draw a rounded box of ``size`` (width, height) at ``centre`` with centred text."""
    (x, y), (width, height) = centre, size
    axes.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0,rounding_size=0.015",
                                  facecolor=fill, edgecolor=edge, linewidth=1.3))
    axes.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.35)


def arrow(axes, start, end, color=LINE, style="-|>", width=1.3, **keywords) -> None:
    """Draw an arrow from ``start`` to ``end``."""
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=color, linewidth=width, **keywords))


def draw_lander(axes) -> None:
    """Panel a: side view of the simulated Apollo LM (collision shapes of the model) on flat ground, to scale."""
    axes.set_xlim(-9.6, 9.4)
    axes.set_ylim(-1.0, 7.0)
    axes.set_aspect("equal")
    # Flat ground with hatching.
    axes.add_patch(Rectangle((-6.2, -0.75), 12.4, 0.75, facecolor="#e8e7e3", edgecolor="none"))
    axes.plot([-6.2, 6.2], [0, 0], color=LINE, linewidth=1.2)
    for x in [-6.0 + 0.45 * i for i in range(27)]:
        axes.plot([x, x - 0.3], [0, -0.3], color=MUTED, linewidth=0.6)
    # Nozzle (no collision shape) and stages.
    nozzle = [(-0.42, DESCENT_BOTTOM), (0.42, DESCENT_BOTTOM), (NOZZLE_EXIT_DIAMETER / 2, NOZZLE_EXIT_HEIGHT), (-NOZZLE_EXIT_DIAMETER / 2, NOZZLE_EXIT_HEIGHT)]
    axes.add_patch(Polygon(nozzle, closed=True, facecolor="#cfcfcb", edgecolor=LINE, linewidth=1.0))
    axes.add_patch(Rectangle((-DESCENT_WIDTH / 2, DESCENT_BOTTOM), DESCENT_WIDTH, DESCENT_HEIGHT, facecolor="#efe3b8", edgecolor=LINE, linewidth=1.2))
    descent_top = DESCENT_BOTTOM + DESCENT_HEIGHT
    axes.add_patch(Rectangle((-ASCENT_DEPTH / 2, descent_top), ASCENT_DEPTH, ASCENT_HEIGHT, facecolor=BODY_FILL, edgecolor=LINE, linewidth=1.2))
    # Legs: a spring-damper strut along the body z axis above each dish footpad, and the leg to the stage.
    for side in (-1, 1):
        x = side * FOOTPAD_RADIUS
        axes.add_patch(Ellipse((x, FOOTPAD_DEPTH / 2), FOOTPAD_DIAMETER, FOOTPAD_DEPTH * 1.4, facecolor="white", edgecolor=INK, linewidth=1.3))
        bottom, top = FOOTPAD_DEPTH, 1.25
        coil_x = [x] + [x + side * 0.16 * (1 if i % 2 else -1) for i in range(1, 8)] + [x]
        coil_z = [bottom + (top - bottom) * i / 8 for i in range(9)]
        axes.plot(coil_x, coil_z, color=INK, linewidth=1.1)
        axes.plot([x, side * DESCENT_WIDTH / 2], [top, DESCENT_BOTTOM + 1.15], color=INK, linewidth=2.0)
        axes.plot([x, side * DESCENT_WIDTH / 2], [top, DESCENT_BOTTOM + 0.05], color=INK, linewidth=1.2)
        # RCS cluster: four small jet nozzles at the cluster point.
        jet_x = side * RCS_CLUSTER_ARM
        for dx, dz in ((0.32, 0), (-0.32, 0), (0, 0.32), (0, -0.32)):
            axes.plot([jet_x, jet_x + dx], [RCS_CLUSTER_HEIGHT, RCS_CLUSTER_HEIGHT + dz], color=LINE, linewidth=2.2, solid_capstyle="butt")
        axes.add_patch(Rectangle((jet_x - 0.11, RCS_CLUSTER_HEIGHT - 0.11), 0.22, 0.22, facecolor="#9b9a95", edgecolor=LINE, linewidth=0.8))
    # Body axis, centre of mass, thrust and weight.
    axes.plot([0, 0], [-0.2, 6.6], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axes.text(0.12, 6.62, "body z (thrust axis)", fontsize=8.5, color=MUTED, va="bottom")
    arrow(axes, (0.0, NOZZLE_EXIT_HEIGHT - 0.05), (0.0, -0.75), color=ANALYTIC, width=0.0, style="-")
    axes.add_patch(Polygon([(-0.55, NOZZLE_EXIT_HEIGHT), (0.55, NOZZLE_EXIT_HEIGHT), (0.0, NOZZLE_EXIT_HEIGHT - 0.42)], closed=True, facecolor="#f6c7a6", edgecolor="none"))
    arrow(axes, (0.45, CENTRE_OF_MASS_HEIGHT), (0.45, CENTRE_OF_MASS_HEIGHT + 1.7), color=ANALYTIC, width=2.2)
    axes.text(0.62, CENTRE_OF_MASS_HEIGHT + 1.45, "T", fontsize=11, color=ANALYTIC, fontweight="bold")
    arrow(axes, (-0.45, CENTRE_OF_MASS_HEIGHT), (-0.45, CENTRE_OF_MASS_HEIGHT - 1.5), color=SECONDARY, width=2.2)
    axes.text(-1.75, CENTRE_OF_MASS_HEIGHT - 1.2, "m g", fontsize=11, color=SECONDARY, fontweight="bold")
    axes.add_patch(Ellipse((0, CENTRE_OF_MASS_HEIGHT), 0.32, 0.32, facecolor="white", edgecolor=INK, linewidth=1.2))
    axes.add_patch(Polygon([(0, CENTRE_OF_MASS_HEIGHT), (0.16, CENTRE_OF_MASS_HEIGHT), (0, CENTRE_OF_MASS_HEIGHT + 0.16)], facecolor=INK))
    axes.add_patch(Polygon([(0, CENTRE_OF_MASS_HEIGHT), (-0.16, CENTRE_OF_MASS_HEIGHT), (0, CENTRE_OF_MASS_HEIGHT - 0.16)], facecolor=INK))
    # Labels on the left.
    label(axes, "ascent stage (box)", (-1.4, 5.4), (-9.4, 6.3))
    label(axes, "descent stage\n(cylinder, Ø 4.22 m)", (-1.85, 2.95), (-9.4, 4.6))
    label(axes, "spring-damper strut along body z\nstroke 0.81 m, k = 78.8 kN/m,\ncritically damped", (-FOOTPAD_RADIUS - 0.15, 0.75), (-9.4, 2.5))
    label(axes, "dish footpad, Ø 0.94 m", (-FOOTPAD_RADIUS - 0.4, 0.1), (-9.4, 0.75))
    # Labels on the right.
    label(axes, "RCS cluster: 4 jets × 445 N\nmin. pulse 14 ms,\nvalve delay 7 ms", (RCS_CLUSTER_ARM + 0.35, RCS_CLUSTER_HEIGHT), (4.2, 6.1))
    label(axes, "centre of mass\n7,327 kg at start\n(349 kg propellant)", (0.16, CENTRE_OF_MASS_HEIGHT - 0.05), (4.6, 3.55))
    label(axes, "descent engine\nT = 10-65 % of 46.7 kN\nlag 0.08 s, slew 85 %/s,\ntail-off 0.5 s", (0.62, 0.95), (5.0, 1.75))
    axes.text(-6.0, -0.55, "flat ground, Coulomb friction μ = 0.4, lunar gravity 1.62 m/s²", fontsize=9, color=SECONDARY, va="center")


def draw_loop(axes) -> None:
    """Panel b: the closed loop shared by both simulators; only the plant differs."""
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    width = 0.70
    box(axes, (0.55, 0.92), (width, 0.12), "PPO policy (MLP), 50 Hz\nthrottle + thrust-direction tilt (≤ 15°)", "white", LINE)
    box(axes, (0.55, 0.74), (width, 0.12), "attitude controller (shared with ZEM/ZEV)\ntilt command → RCS jet duties", "white", LINE)
    box(axes, (0.55, 0.555), (width, 0.13), "actuators: engine lag, slew limit, tail-off;\nRCS minimum pulse and valve delay", "white", LINE)
    box(axes, (0.365, 0.315), (0.32, 0.21), "analytic plant\npure PyTorch, 200 Hz\n6-DOF rigid body,\nanalytic spring legs", "#fbe3d4", ANALYTIC, fontsize=9)
    box(axes, (0.735, 0.315), (0.32, 0.21), "PhysX plant\nIsaac Lab, 200 Hz\narticulation, legs on\nprismatic spring joints", "#dce9f8", PHYSX, fontsize=9)
    axes.text(0.55, 0.315, "or", ha="center", va="center", fontsize=10, color=SECONDARY, fontweight="bold")
    box(axes, (0.55, 0.08), (width, 0.11), "touchdown metric + reward\n(same rule and limits in both)", "white", LINE)
    for top, bottom in ((0.86, 0.80), (0.68, 0.62)):
        arrow(axes, (0.55, top), (0.55, bottom))
    arrow(axes, (0.50, 0.49), (0.40, 0.42))
    arrow(axes, (0.60, 0.49), (0.70, 0.42))
    arrow(axes, (0.40, 0.21), (0.50, 0.135))
    arrow(axes, (0.70, 0.21), (0.60, 0.135))
    # Feedback: the true state goes back to the policy.
    axes.plot([0.2, 0.11, 0.11], [0.08, 0.08, 0.92], color=MUTED, linewidth=1.1)
    arrow(axes, (0.11, 0.92), (0.2, 0.92), color=MUTED, width=1.1)
    axes.text(0.075, 0.5, "true state (16 values)", rotation=90, ha="center", va="center", fontsize=8.5, color=MUTED)


def draw_checks(axes) -> None:
    """Panel c: one episode, from the start box to the success rule."""
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    height, y = 0.52, 0.62
    boxes = [
        (0.115, 0.205, "Start (S1 box)\naltitude 5-50 m\n$v_z$ -6 to 0 m/s, $v_x$, $v_y$ ±2 m/s\ntilt 0-10°", FLIGHT_FILL, FLIGHT_EDGE),
        (0.355, 0.2, "Powered descent\npolicy flies until a\nfootpad force > 100 N", FLIGHT_FILL, FLIGHT_EDGE),
        (0.605, 0.225, "First contact\n(values one step before)\n$v_z$ ≤ 3.05 m/s, $v_h$ ≤ 1.22 m/s*\ntilt ≤ 6°, tilt rate ≤ 2°/s", CHECK_FILL, CHECK_EDGE),
        (0.865, 0.225, "Settle (≤ 10 s)\nspeed < 0.02 m/s and\nrate < 0.5°/s for 1 s;\nno tip-over, no body strike", CHECK_FILL, CHECK_EDGE),
    ]
    for x, width, text, fill, edge in boxes:
        box(axes, (x, y), (width, height), text, fill, edge, fontsize=9.5)
    for (x0, w0, *_), (x1, w1, *_) in zip(boxes, boxes[1:]):
        arrow(axes, (x0 + w0 / 2 + 0.005, y), (x1 - w1 / 2 - 0.005, y))
    axes.text(0.865, 0.27, "→ success", ha="center", fontsize=10, color=CHECK_EDGE, fontweight="bold")
    axes.text(0.355, 0.24, "fails: out of bounds (300 m / 200 m),\npropellant depleted, time limit 90 s", ha="center", va="center", fontsize=8.8, color=NOTE)
    axes.text(0.605, 0.24, "T8 ablation steps b1-b5 train with the\nepisode ending here (no settle)", ha="center", va="center", fontsize=8.8, color=NOTE)
    axes.text(0.01, 0.05, "* above 2.13 m/s vertical the horizontal limit falls linearly to 0 at 3.05 m/s; limits from the LM landing-gear "
              "design envelope (NASA TN D-6850). Held-out evaluation: the same 4,096 starts for every policy.",
              fontsize=8.3, color=SECONDARY)


def main() -> None:
    """Draw the three panels and save the figure as PNG and PDF."""
    figure = plt.figure(figsize=(13.6, 8.6), facecolor="white")
    draw_lander(panel(figure, (0.015, 0.355, 0.585, 0.63), "a  Lander model (Apollo LM preset, to scale)"))
    draw_loop(panel(figure, (0.615, 0.355, 0.37, 0.63), "b  Same loop, two plants"))
    draw_checks(panel(figure, (0.015, 0.015, 0.97, 0.325), "c  One episode and its checks"))
    for suffix in ("png", "pdf"):
        path = ROOT / "plots" / f"method_overview.{suffix}"
        figure.savefig(path, dpi=200, facecolor="white")
        print(f"wrote {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
