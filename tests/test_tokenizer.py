import numpy as np

from trajectory_token_sieve.tokenizer import TrajectoryTokenizer


def test_tokenization_round_trip_tokens():
    tokenizer = TrajectoryTokenizer()
    states = np.asarray([-0.2, 0.0, 0.3, 0.7])
    actions = np.asarray([0.1, 0.4, -0.2, 0.8])
    rewards = np.asarray([1.0, 1.2, 0.7, 1.5])
    tokens = tokenizer.encode(states, actions, rewards)
    decoded = tokenizer.decode(tokens)
    retokens = tokenizer.encode(decoded.states, decoded.actions, decoded.rewards)
    assert retokens == tokens
    assert len(tokens) == 12
