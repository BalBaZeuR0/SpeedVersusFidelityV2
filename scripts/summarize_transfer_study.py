# SPDX-License-Identifier: BSD-3-Clause
"""Print the transfer-study summary (tests T2, T3, T5, T7, T8) from the result files in ``results/``.

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


def summarize_time_to_success_in_physx() -> None:
    """Print test T7: learning wall-clock time until the checkpoints reach 50, 90 and 95 % success in PhysX.

    Every 25th checkpoint of the 20 T5 trainings was flown in PhysX on the held-out starts; the time
    is that of the first checkpoint at or above the threshold (a 25-iteration grid: about 97 s in
    PhysX, 33 s in the analytic simulator).
    """
    checkpoints = pandas.read_csv(RESULTS / "t7_success_in_physx_by_checkpoint.csv")
    checkpoints = checkpoints.sort_values(["trained_in", "seed", "iteration"])
    print("T7: checkpoints flown in PhysX, learning time to reach a success rate in PhysX (10 seeds)")
    for threshold in (0.5, 0.9, 0.95):
        times = {}
        for simulator in SIMULATORS:
            rows = checkpoints[checkpoints.trained_in == simulator]
            times[simulator] = numpy.array(
                [group[group.success_in_physx >= threshold].learn_wall_clock_s.iloc[0] for _, group in rows.groupby("seed")]
            )
        generator = numpy.random.default_rng(0)
        ratios = [
            generator.choice(times["physx"], 10).mean() / generator.choice(times["analytic"], 10).mean()
            for _ in range(10000)
        ]
        welch = stats.ttest_ind(times["physx"], times["analytic"], equal_var=False)
        print(
            f"  {100 * threshold:3.0f} %: PhysX-trained {times['physx'].mean():4.0f} s, analytic-trained "
            f"{times['analytic'].mean():4.0f} s, ratio {times['physx'].mean() / times['analytic'].mean():.2f} "
            f"(bootstrap 95 % CI {numpy.percentile(ratios, 2.5):.2f} to {numpy.percentile(ratios, 97.5):.2f}), "
            f"Welch p {format_p_value(welch.pvalue)}"
        )


ABLATION_STEPS = {
    "b1_contact": "b1 episode ends at first contact",
    "b2_ideal": "b2 b1 + ideal actuators",
    "b3_point_mass": "b3 point mass, ideal actuators",
    "b4_point_mass_actuators": "b4 point mass, real actuators",
    "b5_point_mass_lag": "b5 point mass, 1.3 s attitude lag",
}


def short_interval(values: numpy.ndarray) -> str:
    """Return the mean and the 95 % t interval over seeds, in percent, one decimal."""
    mean = values.mean()
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / numpy.sqrt(len(values))
    low, high = max(0.0, mean - half_width), min(1.0, mean + half_width)  # a rate stays within 0-100 %
    return f"{100 * mean:5.1f} % (95 % CI {100 * low:5.1f} to {100 * high:5.1f})"


def summarize_ablation(success: pandas.DataFrame) -> None:
    """Print test T8: PhysX success of policies trained in ever simpler analytic models (5 seeds each).

    "Judged at first contact" counts a not_settled landing (inside every limit at contact, still
    moving when the settle time ran out) as a success.
    """
    ablation = pandas.read_csv(RESULTS / "t8_ablation_in_physx.csv")
    analytic_in_physx = success[(success.trained_in == "analytic") & (success.flown_in == "physx") & (success.seed < 5)]
    full = analytic_in_physx.success_rate.to_numpy()
    print("T8: trained in simpler analytic models, flown in PhysX (5 seeds); standard rule | judged at first contact")
    print(f"  {'full analytic model (T5 seeds 0-4)':35s} {short_interval(full)} | {short_interval(full)}")
    for step, name in ABLATION_STEPS.items():
        rows = ablation[ablation.step == step]
        standard, at_contact = rows.success.to_numpy(), rows.success_judged_at_first_contact.to_numpy()
        print(f"  {name:35s} {short_interval(standard)} | {short_interval(at_contact)}")


if __name__ == "__main__":
    t5_success = pandas.read_csv(RESULTS / "t5_success_by_seed.csv")
    summarize_transfer(t5_success)
    summarize_training_time()
    summarize_time_to_success_in_physx()
    summarize_ablation(t5_success)
    summarize_agreement_and_speed()
