import numpy as np

from trajectory_token_sieve.environment import generate_offline_dataset
from trajectory_token_sieve.model import SmoothedAutoregressiveTT
from trajectory_token_sieve.tokenizer import TrajectoryTokenizer


def test_autoregressive_probability_normalization():
    tokenizer = TrajectoryTokenizer()
    dataset = generate_offline_dataset(80, horizon=5, seed=3)
    tokens = [tokenizer.encode(t.states, t.actions, t.rewards) for t in dataset]
    model = SmoothedAutoregressiveTT(tokenizer, context=4, alpha=0.05).fit(tokens)
    prefixes = [[], tokens[0][:1], tokens[0][:2], tokens[0][:7]]
    for prefix in prefixes:
        probs = model.next_probs(prefix)
        assert np.all(probs >= 0.0)
        assert np.isclose(probs.sum(), 1.0)
