from wsd.entropy import concentration, normalized_entropy, normalized_entropy_with_support, shannon_entropy


def test_shannon_entropy_zero_for_constant_sequence() -> None:
    assert shannon_entropy(["a", "a", "a"]) == 0.0
    assert normalized_entropy(["a", "a", "a"]) == 0.0


def test_normalized_entropy_in_unit_interval() -> None:
    value = normalized_entropy(["a", "b", "c", "d"])
    assert 0.0 <= value <= 1.0
    assert round(value, 6) == 1.0


def test_normalized_entropy_with_support_handles_unused_branches() -> None:
    value = normalized_entropy_with_support([3, 1], support_size=4)
    assert 0.0 <= value <= 1.0


def test_concentration_increases_for_repeated_sequence() -> None:
    assert concentration(["a", "a", "a", "a"]) > concentration(["a", "b", "c", "d"])
