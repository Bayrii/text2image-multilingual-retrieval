from src.evaluation.metrics import (compute_all_metrics,
                                    mean_average_precision, recall_at_k)


def test_recall_at_1_perfect():
    gts = [10, 20, 30]
    retr = [[10, 99], [20, 5], [30, 1]]
    assert recall_at_k(gts, retr, 1) == 1.0


def test_recall_at_1_zero():
    gts = [10, 20, 30]
    retr = [[1, 10], [2, 20], [3, 30]]
    assert recall_at_k(gts, retr, 1) == 0.0
    assert recall_at_k(gts, retr, 2) == 1.0


def test_recall_partial():
    gts = [1, 2, 3, 4]
    retr = [[1, 0], [0, 0], [3, 0], [0, 0]]
    assert recall_at_k(gts, retr, 1) == 0.5


def test_map_correct():
    # gt at rank 1 -> AP=1.0; gt at rank 2 -> AP=0.5; gt missing -> 0
    assert abs(mean_average_precision([1], [[1, 2, 3]], 3) - 1.0) < 1e-9
    assert abs(mean_average_precision([1], [[2, 1, 3]], 3) - 0.5) < 1e-9
    assert mean_average_precision([1], [[2, 3, 4]], 3) == 0.0


def test_compute_all_metrics_per_language():
    res = {
        "en": [(1, [1, 2, 3]), (2, [3, 2, 1])],
        "fr": [(1, [9, 9, 1]), (2, [9, 9, 9])],
    }
    out = compute_all_metrics(res, recall_ks=[1, 5], map_k=5)
    assert out["en"]["R@1"] == 0.5
    assert out["en"]["R@5"] == 1.0
    assert out["fr"]["R@1"] == 0.0
    assert out["overall"]["n_queries"] == 4
