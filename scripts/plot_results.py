# SPDX-License-Identifier: BSD-3-Clause
"""Draw the README figures from the result files in ``results/``.

Usage::

    python scripts/plot_results.py

Writes ``plots/t5_transfer_success.png``, ``plots/training_curves.png``,
``plots/t2_throughput.png``, ``plots/t7_success_in_physx.png``, ``plots/t8_model_ablation.png``
and the README header image ``assets/hero.png``.
Needs only numpy, pandas and matplotlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Colour follows the simulator in every figure (categorical slots 1 and 2, checked for colour-vision deficiency).
COLOUR = {"physx": "#2a78d6", "analytic": "#eb6834"}
NAME = {"physx": "PhysX (rigid body, Isaac Lab)", "analytic": "Analytic (pure torch)"}
SURFACE, INK, SECONDARY, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style_axes(axes) -> None:
    """Recessive grid and axes on the chart surface."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=SECONDARY, labelsize=9)
    axes.xaxis.label.set_color(SECONDARY)
    axes.yaxis.label.set_color(SECONDARY)
    axes.title.set_color(INK)


def draw_transfer_success(axes) -> None:
    """Test T5: success of every (trained in, flown in) pair; seeds as dots, mean with 95 % t interval."""
    success = pandas.read_csv(RESULTS / "t5_success_by_seed.csv")
    groups = [("physx", "physx"), ("analytic", "physx"), ("physx", "analytic"), ("analytic", "analytic")]
    jitter = numpy.random.default_rng(1).uniform(-0.09, 0.09, 10)
    for position, (trained, flown) in enumerate(groups):
        values = 100 * success[(success.trained_in == trained) & (success.flown_in == flown)].sort_values("seed").success_rate.to_numpy()
        mean = values.mean()
        half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / numpy.sqrt(len(values))
        axes.scatter(position + jitter, values, s=22, color=COLOUR[trained], alpha=0.45, edgecolor=SURFACE, linewidth=1, zorder=3)
        axes.errorbar(position + 0.28, mean, yerr=half_width, fmt="o", color=COLOUR[trained], markersize=7,
                      elinewidth=2, capsize=4, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        axes.text(position + 0.36, mean, f"{mean:.2f} %", va="center", fontsize=8.5, color=INK)
    axes.set_xticks(range(4), ["PhysX-\ntrained", "analytic-\ntrained", "PhysX-\ntrained", "analytic-\ntrained"])
    axes.set_xlim(-0.4, 3.85)
    axes.set_ylim(96.5, 98.8)
    axes.set_ylabel("Landing success on 4,096 held-out starts (%)")
    for centre, label in ((0.5, "flown in PhysX"), (2.5, "flown in analytic")):
        axes.text(centre, 98.72, label, ha="center", fontsize=9, color=SECONDARY)
    axes.axvline(1.5, color=BASELINE, linewidth=0.8)
    axes.set_title("Analytic-trained policies land as well in PhysX\n(10 seeds; dots = seeds, bar = mean and 95 % CI)", fontsize=10.5, loc="left")
    style_axes(axes)


def draw_training_curves(axes) -> None:
    """Training success against learning wall-clock time: median over 10 seeds with the interquartile band."""
    curves = pandas.read_csv(RESULTS / "training_curves.csv")
    for simulator in ("physx", "analytic"):
        run = curves[curves.simulator == simulator].pivot_table(index="iteration", columns="seed", values="training_success_rate")
        clock = curves[curves.simulator == simulator].groupby("iteration").learn_wall_clock_s.median() / 60.0
        median, low, high = (100 * run.quantile(q, axis=1) for q in (0.5, 0.25, 0.75))
        axes.fill_between(clock.loc[run.index], low, high, color=COLOUR[simulator], alpha=0.18, linewidth=0)
        axes.plot(clock.loc[run.index], median, color=COLOUR[simulator], linewidth=2, label=NAME[simulator])
        # Mean over seeds of the first iteration at or above 90 % (the number in the README and the summary).
        rows = curves[curves.simulator == simulator].sort_values("iteration")
        first = [group[group.training_success_rate >= 0.9].learn_wall_clock_s.iloc[0] for _, group in rows.groupby("seed")]
        reached = numpy.mean(first) / 60.0
        axes.plot([reached], [90], "o", color=COLOUR[simulator], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        label_offset = (8, -14) if simulator == "analytic" else (8, -32)
        axes.annotate(f"90 % after {reached:.1f} min", (reached, 90), xytext=label_offset, textcoords="offset points", fontsize=8.5, color=INK)
    axes.axhline(90, color=BASELINE, linewidth=0.8, linestyle="--")
    axes.set_xlabel("Learning wall-clock time (min), RTX 5060, 4,096 environments")
    axes.set_ylabel("Training success rate (%)")
    axes.set_xlim(0, 40)
    axes.set_ylim(0, 100)
    axes.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    axes.set_title("Same learning curve, about 3x less wall-clock time\n(median of 10 seeds, band = interquartile range; dot = mean time)", fontsize=10.5, loc="left")
    style_axes(axes)


def draw_throughput(axes) -> None:
    """Test T2: environment steps per second against the number of environments (log-log), PPO policy in the loop."""
    rows = [json.loads(line) for line in (RESULTS / "t2_throughput.jsonl").read_text().splitlines()]
    frame = pandas.DataFrame(rows)
    frame = frame[frame.agent == "ppo_mlp"]
    for simulator, key in (("physx", "isaac_physx"), ("analytic", "analytic")):
        grouped = frame[frame.simulator == key].groupby("num_envs").env_steps_per_second
        median = grouped.median()
        axes.fill_between(median.index, grouped.min(), grouped.max(), color=COLOUR[simulator], alpha=0.18, linewidth=0)
        axes.plot(median.index, median, "o-", color=COLOUR[simulator], linewidth=2, markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=NAME[simulator])
        axes.annotate(f"{median.iloc[-1] / 1e6:.2f} M", (median.index[-1], median.iloc[-1]), xytext=(6, -3),
                      textcoords="offset points", fontsize=8.5, color=INK)
    axes.set_xscale("log", base=2)
    counts = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    axes.set_xticks(counts, [f"{count:,}" for count in counts])
    axes.minorticks_off()
    axes.set_yscale("log")
    axes.set_xlabel("Number of parallel environments")
    axes.set_ylabel("Environment steps per second")
    axes.set_xlim(0.7, 40000)
    axes.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    axes.set_title("Throughput, PPO policy in the loop: 1.6x at 1 env, 9.4x at 16,384\n(median of 5 repeats, band = min to max)", fontsize=10.5, loc="left")
    style_axes(axes)


def draw_success_in_physx_during_training(axes) -> None:
    """Test T7: every 25th checkpoint flown in PhysX, success against learning wall-clock time (median and IQR of 10 seeds)."""
    checkpoints = pandas.read_csv(RESULTS / "t7_success_in_physx_by_checkpoint.csv")
    for simulator in ("physx", "analytic"):
        rows = checkpoints[checkpoints.trained_in == simulator]
        success = rows.pivot_table(index="iteration", columns="seed", values="success_in_physx")
        clock = rows.groupby("iteration").learn_wall_clock_s.median() / 60.0
        median, low, high = (100 * success.quantile(q, axis=1) for q in (0.5, 0.25, 0.75))
        axes.fill_between(clock.loc[success.index], low, high, color=COLOUR[simulator], alpha=0.18, linewidth=0)
        axes.plot(clock.loc[success.index], median, color=COLOUR[simulator], linewidth=2, label=f"trained in {NAME[simulator]}")
        # Mean over seeds of the first checkpoint at or above 90 % in PhysX.
        first = [group[group.success_in_physx >= 0.9].learn_wall_clock_s.iloc[0] for _, group in rows.sort_values("iteration").groupby("seed")]
        reached = numpy.mean(first) / 60.0
        axes.plot([reached], [90], "o", color=COLOUR[simulator], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        label_offset = (8, -14) if simulator == "analytic" else (8, -32)
        axes.annotate(f"90 % in PhysX after {reached:.1f} min", (reached, 90), xytext=label_offset, textcoords="offset points", fontsize=8.5, color=INK)
    axes.axhline(90, color=BASELINE, linewidth=0.8, linestyle="--")
    axes.set_xlabel("Learning wall-clock time (min), RTX 5060, 4,096 environments")
    axes.set_ylabel("Landing success in PhysX (%)")
    axes.set_xlim(0, 40)
    axes.set_ylim(0, 100)
    axes.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    axes.set_title("Checkpoints flown in PhysX: 90 % reached 3.3x sooner\n(median of 10 seeds, band = interquartile range; dot = mean time)", fontsize=10.5, loc="left")
    style_axes(axes)


ABLATION_STEPS = [
    ("full", "Full analytic\nmodel (T5)"),
    ("b1_contact", "Ends at\nfirst contact"),
    ("b2_ideal", "+ ideal\nactuators"),
    ("b3_point_mass", "Point mass,\nideal actuators"),
    ("b4_point_mass_actuators", "Point mass,\nreal actuators"),
    ("b5_point_mass_lag", "Point mass,\nattitude lag"),
]


def draw_model_ablation(axes) -> None:
    """Test T8: PhysX success of policies trained in ever simpler analytic models; seeds as dots, mean as a marker.

    Filled: the standard success rule. Hollow: judged at the first contact (a landing that is inside
    every limit at contact but still moving when the settle time runs out counts as a success).
    """
    ablation = pandas.read_csv(RESULTS / "t8_ablation_in_physx.csv")
    full = pandas.read_csv(RESULTS / "t5_success_by_seed.csv")
    full = full[(full.trained_in == "analytic") & (full.flown_in == "physx") & (full.seed < 5)]
    # T5 policies have no not_settled outcome on these starts, so both rules give the same value.
    values = {"full": (full.success_rate.to_numpy(), full.success_rate.to_numpy())}
    for step, _ in ABLATION_STEPS[1:]:
        rows = ablation[ablation.step == step]
        if len(rows):
            values[step] = (rows.success.to_numpy(), rows.success_judged_at_first_contact.to_numpy())
    steps = [(step, label) for step, label in ABLATION_STEPS if step in values]
    jitter = numpy.random.default_rng(2).uniform(-0.07, 0.07, 5)
    colour = COLOUR["analytic"]
    for position, (step, _) in enumerate(steps):
        standard, at_contact = (100 * v for v in values[step])
        axes.scatter(position - 0.14 + jitter[: len(standard)], standard, s=18, color=colour, alpha=0.4, edgecolor=SURFACE, linewidth=1, zorder=3)
        axes.scatter(position + 0.14 + jitter[: len(at_contact)], at_contact, s=18, facecolor="none", edgecolor=colour, alpha=0.6, linewidth=1, zorder=3)
        axes.plot(position - 0.14, standard.mean(), "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        axes.plot(position + 0.14, at_contact.mean(), "o", markerfacecolor=SURFACE, markeredgecolor=colour, markersize=8, markeredgewidth=2, zorder=4)
        axes.text(position, max(standard.mean(), at_contact.mean()) + 4, f"{standard.mean():.0f} | {at_contact.mean():.0f} %", ha="center", fontsize=8.5, color=INK)
    axes.plot([], [], "o", color=colour, markersize=7, label="standard success rule")
    axes.plot([], [], "o", markerfacecolor=SURFACE, markeredgecolor=colour, markersize=7, markeredgewidth=2, label="judged at first contact")
    axes.set_xticks(range(len(steps)), [label for _, label in steps], fontsize=8.5)
    axes.set_xlim(-0.6, len(steps) - 0.4)
    axes.set_ylim(-3, 108)
    axes.set_ylabel("Landing success in PhysX (%)")
    axes.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    axes.set_title("Which simplification breaks the transfer?\n(trained in the analytic model, flown in PhysX; 5 seeds, dots = seeds, marker = mean)", fontsize=10.5, loc="left")
    style_axes(axes)


def save(figure, path: Path) -> None:
    """Save a figure on the page colour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


def main() -> None:
    """Draw every figure."""
    for name, draw, size in (
        ("plots/t5_transfer_success.png", draw_transfer_success, (6.4, 4.4)),
        ("plots/training_curves.png", draw_training_curves, (6.4, 4.4)),
        ("plots/t2_throughput.png", draw_throughput, (6.4, 4.4)),
        ("plots/t7_success_in_physx.png", draw_success_in_physx_during_training, (6.4, 4.4)),
        ("plots/t8_model_ablation.png", draw_model_ablation, (8.0, 4.6)),
    ):
        figure, axes = plt.subplots(figsize=size, facecolor=SURFACE)
        draw(axes)
        save(figure, ROOT / name)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.6), facecolor=SURFACE)
    draw_success_in_physx_during_training(left)
    draw_transfer_success(right)
    figure.tight_layout(w_pad=3)
    save(figure, ROOT / "assets/hero.png")


if __name__ == "__main__":
    main()
