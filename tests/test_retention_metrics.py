from retention_rl.retention.metrics import capability_gap, retention_deficit


def test_retention_deficit_positive_when_support_drops():
    assert retention_deficit(insert_nll=1.0, current_nll=1.7) == 0.7


def test_retention_deficit_is_clipped_at_zero():
    assert retention_deficit(insert_nll=1.7, current_nll=1.0) == 0.0


def test_capability_gap():
    assert capability_gap(stored_return=900, revisit_mean_return=650) == 250
