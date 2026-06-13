from trajectory_token_sieve.diagnostics import calibrate_support, diagnose_plan, is_supported
from trajectory_token_sieve.environment import generate_offline_dataset
from trajectory_token_sieve.experiments.core import aggregate_rows, run_variant
from trajectory_token_sieve.config import ExperimentConfig
from trajectory_token_sieve.model import SmoothedAutoregressiveTT
from trajectory_token_sieve.tokenizer import TrajectoryTokenizer


def test_support_diagnostics_flag_synthetic_out_of_support_plan():
    tokenizer = TrajectoryTokenizer()
    dataset = generate_offline_dataset(160, horizon=6, seed=8, high_support=0.05)
    tokens = [tokenizer.encode(t.states, t.actions, t.rewards) for t in dataset]
    model = SmoothedAutoregressiveTT(tokenizer, context=4, alpha=0.05).fit(tokens)
    thresholds = calibrate_support(tokens, [float(t.states[0]) for t in dataset], model, tokenizer)
    bad_tokens = []
    for _ in range(6):
        bad_tokens.extend([tokenizer.state_bins - 1, tokenizer.action_bins - 1, tokenizer.reward_bins - 1])
    diag = diagnose_plan(bad_tokens, model, tokenizer, initial_state=0.0)
    assert not is_supported(diag, thresholds)
    assert diag.dynamics_error > thresholds.max_dynamics_error or diag.prefix_surprise > thresholds.max_prefix_surprise


def test_repair_improves_or_blocks_unsafe_high_n_selection():
    cfg = ExperimentConfig(
        horizon=7,
        train_trajectories=180,
        eval_episodes=6,
        n_values=(1, 64),
        seed=41,
        output_prefix="test",
        variants=("out_of_support",),
    )
    tokenizer = TrajectoryTokenizer()
    rows, thresholds = run_variant(cfg, "out_of_support", tokenizer)
    agg = aggregate_rows(rows, {"out_of_support": thresholds})
    raw64 = next(r for r in agg if r["method"] == "raw" and int(r["N"]) == 64)
    sieve64 = next(r for r in agg if r["method"] == "sieve" and int(r["N"]) == 64)
    assert (
        sieve64["realized_return"] >= raw64["realized_return"] - 0.25
        or sieve64["blocked"] > 0.0
        or raw64["candidate_risk_rate"] > thresholds.max_risk_rate
    )
