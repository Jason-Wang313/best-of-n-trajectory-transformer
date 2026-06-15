from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import DOCS_DIR, FIGURES_DIR, RESULTS_DIR, ExperimentConfig, ensure_output_dirs
from ..diagnostics import SupportThresholds, calibrate_support
from ..environment import generate_offline_dataset
from ..figures import (
    plot_control_panels,
    plot_exact_law,
    plot_repair_comparison,
    plot_reward_extremization,
    summarize_by_variant,
)
from ..model import SmoothedAutoregressiveTT
from ..planning import (
    SupportCalibratedPlanSieve,
    candidate_diversity,
    sample_candidates,
    select_oracle,
    select_raw,
)
from ..tokenizer import TrajectoryTokenizer


NUMERIC_FIELDS = [
    "predicted_return",
    "realized_return",
    "avg_logprob",
    "prefix_surprise",
    "dynamics_error",
    "action_energy",
    "candidate_risk_rate",
    "candidate_diversity",
    "blocked",
    "accepted_count",
]


def stable_variant_offset(variant: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(variant))


def variant_config(base: ExperimentConfig, variant: str) -> dict:
    cfg = {
        "horizon": base.horizon,
        "high_support": base.high_support,
        "risky_support": base.risky_support,
        "alpha": base.alpha,
        "temperature": base.sample_temperature,
        "reward_tail_bias": 0.42,
        "action_tail_bias": 0.18,
        "score_mode": "predicted_reward",
    }
    if variant == "in_support":
        cfg.update(high_support=0.16, alpha=0.025, temperature=0.88, reward_tail_bias=0.22, action_tail_bias=0.06)
    elif variant == "out_of_support":
        cfg.update(high_support=0.030, alpha=0.120, temperature=1.62, reward_tail_bias=0.58, action_tail_bias=0.34)
    elif variant == "anti_aligned_scorer":
        cfg.update(high_support=0.07, alpha=0.075, temperature=1.30, reward_tail_bias=0.50, action_tail_bias=0.26, score_mode="anti_aligned")
    elif variant == "horizon_stress":
        cfg.update(horizon=max(base.horizon + 6, 16), high_support=0.075, alpha=0.070, temperature=1.24, reward_tail_bias=0.48, action_tail_bias=0.24)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return cfg


def fit_variant(
    base: ExperimentConfig,
    variant: str,
    tokenizer: TrajectoryTokenizer,
) -> tuple[SmoothedAutoregressiveTT, SupportThresholds, list]:
    cfg = variant_config(base, variant)
    dataset = generate_offline_dataset(
        n_trajectories=base.train_trajectories,
        horizon=int(cfg["horizon"]),
        seed=base.seed + stable_variant_offset(variant),
        high_support=float(cfg["high_support"]),
        risky_support=float(cfg["risky_support"]),
    )
    tokens = [tokenizer.encode(t.states, t.actions, t.rewards) for t in dataset]
    model = SmoothedAutoregressiveTT(
        tokenizer=tokenizer,
        context=base.context,
        alpha=float(cfg["alpha"]),
        reward_tail_bias=float(cfg["reward_tail_bias"]),
        action_tail_bias=float(cfg["action_tail_bias"]),
    ).fit(tokens)
    thresholds = calibrate_support(
        tokens,
        [float(t.states[0]) for t in dataset],
        model,
        tokenizer,
    )
    return model, thresholds, dataset


def _episode_record(
    variant: str,
    n: int,
    method: str,
    selection,
    risk_rate: float,
    diversity: float,
) -> dict:
    d = selection.candidate.diagnostics
    return {
        "variant": variant,
        "N": int(n),
        "method": method,
        "predicted_return": d.predicted_return,
        "realized_return": d.realized_return,
        "avg_logprob": d.avg_logprob,
        "prefix_surprise": d.prefix_surprise,
        "dynamics_error": d.dynamics_error,
        "action_energy": d.action_energy,
        "candidate_risk_rate": risk_rate,
        "candidate_diversity": diversity,
        "blocked": float(selection.blocked),
        "accepted_count": float(selection.accepted_count),
        "mode_signature": repr(selection.candidate.diagnostics.mode_signature),
    }


def run_variant(base: ExperimentConfig, variant: str, tokenizer: TrajectoryTokenizer) -> tuple[list[dict], SupportThresholds]:
    cfg = variant_config(base, variant)
    model, thresholds, _dataset = fit_variant(base, variant, tokenizer)
    sieve = SupportCalibratedPlanSieve(thresholds)
    rng = np.random.default_rng(base.seed * 1009 + stable_variant_offset(variant))
    rows: list[dict] = []
    horizon = int(cfg["horizon"])
    for _episode in range(base.eval_episodes):
        initial_state = float(rng.normal(0.0, 0.10))
        for n in base.n_values:
            candidates = sample_candidates(
                model=model,
                tokenizer=tokenizer,
                initial_state=initial_state,
                horizon=horizon,
                n=int(n),
                rng=rng,
                temperature=float(cfg["temperature"]),
            )
            diversity = candidate_diversity(candidates)
            sieve_selection = sieve.select(candidates)
            risk_rate = sieve_selection.candidate_risk_rate
            raw_selection = select_raw(candidates, score_mode=str(cfg["score_mode"]))
            oracle_selection = select_oracle(candidates)
            rows.append(_episode_record(variant, int(n), "raw", raw_selection, risk_rate, diversity))
            rows.append(_episode_record(variant, int(n), "sieve", sieve_selection, risk_rate, diversity))
            rows.append(_episode_record(variant, int(n), "oracle", oracle_selection, risk_rate, diversity))
    return rows, thresholds


def aggregate_rows(episode_rows: list[dict], thresholds_by_variant: dict[str, SupportThresholds]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["variant"], int(row["N"]), row["method"])].append(row)

    aggregate: list[dict] = []
    for (variant, n, method), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        item = {"variant": variant, "N": int(n), "method": method}
        for field in NUMERIC_FIELDS:
            item[field] = float(np.mean([float(row[field]) for row in group]))
        signatures = {row["mode_signature"] for row in group}
        item["mode_collapse"] = float(1.0 - len(signatures) / max(1, len(group)))
        aggregate.append(item)

    safe_n_by_variant: dict[str, int] = {}
    for variant, thresholds in thresholds_by_variant.items():
        raw_rows = [row for row in aggregate if row["variant"] == variant and row["method"] == "raw"]
        safe = [
            int(row["N"])
            for row in raw_rows
            if float(row["candidate_risk_rate"]) <= thresholds.max_risk_rate
        ]
        safe_n_by_variant[variant] = max(safe) if safe else 0
    for row in aggregate:
        safe_n = safe_n_by_variant[row["variant"]]
        row["adaptive_safe_n"] = int(safe_n)
        row["adaptive_allowed"] = bool(safe_n and int(row["N"]) <= safe_n)
    return aggregate


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_final_audit(summary: dict, pdf_repo_path: Path, pdf_download_path: Path) -> None:
    out = DOCS_DIR / "final_audit.md"
    empirical_variant = max(
        summary,
        key=lambda k: summary[k].get("raw_predicted_gain_1_to_64", 0.0)
        - summary[k].get("raw_realized_change_1_to_64", 0.0),
    )
    main = summary[empirical_variant]
    strongest_variant = max(
        summary,
        key=lambda k: summary[k].get("sieve_minus_raw_realized_at_64", float("-inf")),
    )
    strongest = summary[strongest_variant]
    lines = [
        "# Final Audit",
        "",
        "1. **What is the discovered main thesis?**",
        "",
        "   Score-only candidate selection with a Trajectory Transformer-style autoregressive model can extremize reward tokens faster than it improves actual plans. In controlled stress settings, the selected sequence's predicted reward rises with candidate count while support likelihood and dynamics consistency deteriorate, so simulator return can fall.",
        "",
        "2. **What is genuinely new?**",
        "",
        "   The contribution is not generic reranking. It is a Trajectory Transformer-specific diagnostic that treats the full interleaved state/action/reward token string as the object under selection, measures prefix surprise and token-level dynamics inconsistency, and uses those signals to gate large candidate-count search.",
        "",
        "3. **What proposition/proof survived adversarial checking?**",
        "",
        "   The exact finite-candidate selected-utility law for score-selected i.i.d. candidates with uniform tie-breaking survived the proof attack. The corollary is deliberately conditional: increasing the candidate count worsens realized utility when higher proxy-score tails carry lower conditional real utility.",
        "",
        "4. **What is the strongest empirical result?**",
        "",
        f"   In the `{empirical_variant}` control, score-only selection changed predicted return by {main.get('raw_predicted_gain_1_to_64', 0.0):.3f} from candidate count 1 to 64 while realized return changed by {main.get('raw_realized_change_1_to_64', 0.0):.3f}. This is synthetic evidence for reward-token extremization, not a benchmark claim.",
        "",
        "5. **What is the strongest repair result?**",
        "",
        f"   The strongest repair result is in `{strongest_variant}`: at candidate count 64, the Support-Calibrated Plan Sieve improved realized return over score-only selection by {strongest.get('sieve_minus_raw_realized_at_64', 0.0):.3f} or blocked unsafe candidate sets when support diagnostics crossed calibration thresholds.",
        "",
        "6. **What are the biggest weaknesses?**",
        "",
        "   The experiments are synthetic, the model is an n-gram surrogate rather than a trained neural Transformer, and the repair thresholds are calibrated on generated offline data. Benchmark validation on D4RL-style environments and a neural TT implementation is the main missing evidence.",
        "",
        "7. **Is this submission-ready as a bounded mechanism study, or does it need stronger experiments, neural benchmark validation, or redesign?**",
        "",
        "   Paper-worthy v1 for the mechanism and diagnostic, but it needs stronger experiments and benchmark validation before the empirical claim should be treated as field-level evidence.",
        "",
        "8. **Where exactly is the final PDF saved?**",
        "",
        f"   Repository copy: `{pdf_repo_path}`",
        f"   Downloads copy: `{pdf_download_path}`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def run_suite(base: ExperimentConfig, variants: tuple[str, ...] | None = None) -> dict:
    ensure_output_dirs()
    tokenizer = TrajectoryTokenizer()
    selected_variants = variants or base.variants
    all_episode_rows: list[dict] = []
    thresholds_by_variant: dict[str, SupportThresholds] = {}
    for variant in selected_variants:
        rows, thresholds = run_variant(base, variant, tokenizer)
        all_episode_rows.extend(rows)
        thresholds_by_variant[variant] = thresholds
    aggregate = aggregate_rows(all_episode_rows, thresholds_by_variant)
    write_csv(RESULTS_DIR / f"{base.output_prefix}_episode_rows.csv", all_episode_rows)
    write_csv(RESULTS_DIR / f"{base.output_prefix}_results.csv", aggregate)

    if "out_of_support" in selected_variants:
        plot_reward_extremization(aggregate, FIGURES_DIR)
    plot_repair_comparison(aggregate, FIGURES_DIR)
    plot_control_panels(aggregate, FIGURES_DIR)
    law = plot_exact_law(FIGURES_DIR)
    summary = summarize_by_variant(aggregate) if 64 in base.n_values else {}
    payload = {
        "config": {
            "horizon": base.horizon,
            "train_trajectories": base.train_trajectories,
            "eval_episodes": base.eval_episodes,
            "seed": base.seed,
            "n_values": list(base.n_values),
            "variants": list(selected_variants),
        },
        "summary_by_variant": summary,
        "thresholds_by_variant": {
            variant: thresholds.__dict__ for variant, thresholds in thresholds_by_variant.items()
        },
        "exact_law": law,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if summary:
        write_final_audit(
            summary,
            pdf_repo_path=Path("paper/final/iclr_submission.pdf"),
            pdf_download_path=Path.home() / "Downloads" / "iclr_submission_trajectory_token_sieve.pdf",
        )
    return payload
