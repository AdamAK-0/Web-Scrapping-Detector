from wsd.entropy import normalized_entropy, shannon_entropy


def test_shannon_entropy_zero_for_constant_sequence() -> None:
    assert shannon_entropy(["a", "a", "a"]) == 0.0
    assert normalized_entropy(["a", "a", "a"]) == 0.0


def test_normalized_entropy_in_unit_interval() -> None:
    value = normalized_entropy(["a", "b", "c", "d"])
    assert 0.0 <= value <= 1.0
    assert round(value, 6) == 1.0
