# SPDX-License-Identifier: BSD-3-Clause
"""Print the transfer-study summary (tests T2, T3, T5) from the result files in ``results/``.

Usage::

    python scripts/summarize_transfer_study.py

Needs only numpy, pandas and scipy; it reads the exported results and does not need LunarSim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pandas
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIMULATORS = ("physx", "analytic")
EQUIVALENCE_MARGIN = 0.01  # one percentage point of success rate


def mean_with_interval(values: numpy.ndarray) -> str:
    """Return mean, standard deviation and 95 % t interval over seeds, in percent."""
    mean = values.mean()
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / numpy.sqrt(len(values))
    return (
        f"{100 * mean:6.2f} %  sd {100 * values.std(ddof=1):.2f}  "
        f"95 % CI {100 * (mean - half_width):.2f} to {100 * (mean + half_width):.2f}"
    )


def format_p_value(value: float) -> str:
    """Return '= 0.0123' or '< 0.0001' for a p-value."""
    return "< 0.0001" if value < 1e-4 else f"= {value:.4f}"


def summarize_transfer(success: pandas.DataFrame) -> None:
    """Print test T5: success of every (trained in, flown in) pair and the transfer gap in PhysX."""
    seeds = sorted(success["seed"].unique())
    print(f"T5: PPO policies, {len(seeds)} seeds, each flown on the same 4,096 held-out starts")
    for trained in SIMULATORS:
        for flown in SIMULATORS:
            rows = success[(success.trained_in == trained) & (success.flown_in == flown)].sort_values("seed")
            print(f"  trained in {trained:8s} flown in {flown:8s} {mean_with_interval(rows.success_rate.to_numpy())}")
    physx_trained = success[(success.trained_in == "physx") & (success.flown_in == "physx")].success_rate.to_numpy()
    analytic_trained = success[(success.trained_in == "analytic") & (success.flown_in == "physx")].success_rate.to_numpy()
    welch = stats.ttest_ind(analytic_trained, physx_trained, equal_var=False)
    lower = stats.ttest_ind(analytic_trained + EQUIVALENCE_MARGIN, physx_trained, equal_var=False, alternative="greater")
    upper = stats.ttest_ind(analytic_trained - EQUIVALENCE_MARGIN, physx_trained, equal_var=False, alternative="less")
    generator = numpy.random.default_rng(0)
    bootstrap = [
        generator.choice(analytic_trained, len(analytic_trained)).mean()
        - generator.choice(physx_trained, len(physx_trained)).mean()
        for _ in range(20000)
    ]
    print("  transfer gap in PhysX (analytic-trained minus PhysX-trained):")
    print(
        f"    {100 * (analytic_trained.mean() - physx_trained.mean()):+.2f} points, Welch p = {welch.pvalue:.2f}, "
        f"bootstrap 95 % CI {100 * numpy.percentile(bootstrap, 2.5):+.2f} to {100 * numpy.percentile(bootstrap, 97.5):+.2f}, "
        f"equivalence within +-1 point (TOST) p {format_p_value(max(lower.pvalue, upper.pvalue))}"
    )
    agreement = success.drop_duplicates(["trained_in", "seed"]).groupby("trained_in")["same_outcome_code_in_both_simulators"].mean()
    print(
        "  same outcome code when one policy flies the same start in both simulators: "
        + ", ".join(f"{name}-trained {100 * value:.2f} %" for name, value in agreement.items())
    )


def summarize_training_time() -> None:
    """Print the training wall-clock time and the time to 90 % training success in each simulator."""
    curves = pandas.read_csv(RESULTS / "training_curves.csv")
    print("Training (4,096 environments, 600 PPO iterations, RTX 5060):")
    for simulator in SIMULATORS:
        totals, to_ninety = [], []
        for seed in range(10):
            clock = pandas.read_csv(RESULTS / "checkpoint_wall_clock" / f"checkpoint_wall_clock_{simulator}_seed{seed}.csv")
            totals.append(clock.learn_wall_clock_s.iloc[-1])
            run = curves[(curves.simulator == simulator) & (curves.seed == seed)]
            reached = run[run.training_success_rate >= 0.9]
            to_ninety.append(reached.learn_wall_clock_s.iloc[0] if len(reached) else numpy.nan)
        print(
            f"  {simulator:8s} total {numpy.mean(totals):5.0f} s, "
            f"90 % training success after {numpy.nanmean(to_ninety):4.0f} s (mean of 10 seeds)"
        )


def summarize_agreement_and_speed() -> None:
    """Print test T3 (paired ZEM/ZEV flights) and test T2 (environment throughput)."""
    paired = pandas.read_csv(RESULTS / "t3_paired_zem_zev.csv")
    print("T3: ZEM/ZEV, same 4,096 starts in both simulators")
    for row in paired.itertuples():
        print(
            f"  {row.starts:22s} PhysX {100 * row.physx_success_rate:.2f} %  analytic {100 * row.analytic_success_rate:.2f} %  "
            f"same outcome {100 * row.same_outcome_code:.2f} %"
        )
    rows = [json.loads(line) for line in (RESULTS / "t2_throughput.jsonl").read_text().splitlines()]
    throughput = pandas.DataFrame(rows)
    median = throughput.groupby(["agent", "num_envs", "simulator"]).env_steps_per_second.median().unstack()
    print("T2: environment steps per second, median of 5 repeats")
    for (agent, count), values in median.iterrows():
        if count in (1, 1024, 4096, 16384):
            print(
                f"  {agent:8s} {count:6d} envs  PhysX {values['isaac_physx']:9.0f}  analytic {values['analytic']:9.0f}  "
                f"ratio {values['analytic'] / values['isaac_physx']:.1f}"
            )


if __name__ == "__main__":
    summarize_transfer(pandas.read_csv(RESULTS / "t5_success_by_seed.csv"))
    summarize_training_time()
    summarize_agreement_and_speed()
