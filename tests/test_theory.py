from trajectory_token_sieve.theory import (
    expected_selected_utility,
    monte_carlo_selected_utility,
    tail_misalignment_example,
)


def test_score_selected_exact_law_matches_monte_carlo():
    dist = tail_misalignment_example()
    exact = expected_selected_utility(dist, n=8)
    mc = monte_carlo_selected_utility(dist, n=8, trials=60000, seed=5)
    assert abs(exact - mc) < 0.02


def test_tail_misalignment_can_worsen_with_n():
    dist = tail_misalignment_example()
    assert expected_selected_utility(dist, 64) < expected_selected_utility(dist, 1)
