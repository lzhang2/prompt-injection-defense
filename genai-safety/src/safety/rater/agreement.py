"""How well does the rater agree with human labels?"""
from typing import Optional, Sequence


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else float("nan")
    return (observed - expected) / (1 - expected)


def agreement_report(human: Sequence[int],
                     rater: Sequence[Optional[int]]) -> dict:
    """rater entries may be None (failed calls); they are counted, then dropped."""
    pairs = [(h, r) for h, r in zip(human, rater) if r is not None]
    h, r = [p[0] for p in pairs], [p[1] for p in pairs]
    tp = sum(x == 1 and y == 1 for x, y in pairs)
    fp = sum(x == 0 and y == 1 for x, y in pairs)
    fn = sum(x == 1 and y == 0 for x, y in pairs)
    return {
        "n_rated": len(pairs), "n_failed": len(human) - len(pairs),
        "accuracy": sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan"),
        "kappa": cohen_kappa(h, r),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }
