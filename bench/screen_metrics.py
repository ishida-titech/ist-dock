#!/usr/bin/env python3
"""スクリーニング指標（ROC AUC / EF@x% / BEDROC）。lower score = more active-like。

`screen_eval.py` から import して使う。依存は標準ライブラリのみ（sklearn 不使用）。
"""
import math
from bisect import bisect_left, bisect_right


def roc_auc(scores, labels):
    """AUC = P(active のスコア < decoy のスコア) + 0.5·P(=)。lower=active-like。"""
    act = sorted(s for s, l in zip(scores, labels) if l)
    dec = sorted(s for s, l in zip(scores, labels) if not l)
    na, nd = len(act), len(dec)
    if na == 0 or nd == 0:
        return float("nan")
    # 各 active について、スコアがより大きい(=悪い) decoy 数 + 0.5·同点。
    total = 0.0
    for s in act:
        greater = nd - bisect_right(dec, s)
        ties = bisect_right(dec, s) - bisect_left(dec, s)
        total += greater + 0.5 * ties
    return total / (na * nd)


def enrichment_factor(scores, labels, frac):
    """EF@frac: 上位 frac（最良スコア）中の active 濃縮係数。"""
    n = len(scores)
    na = sum(labels)
    if na == 0 or n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    order = sorted(range(n), key=lambda i: scores[i])  # ascending=best first
    top_act = sum(labels[i] for i in order[:k])
    return (top_act / k) / (na / n)


def bedroc(scores, labels, alpha=20.0):
    """BEDROC(α)（Truchon & Bayly 2007）。早期濃縮を強調（α=20 → 上位 ~8% を重視）。"""
    n = len(scores)
    na = sum(labels)
    if na == 0 or na == n:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])  # ascending=best first（rank 1=最良）
    ra = na / n
    s = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            s += math.exp(-alpha * rank / n)
    rie = s / ((ra) * (1 - math.exp(-alpha)) / (math.exp(alpha / n) - 1))
    bedroc = rie * (ra * math.sinh(alpha / 2)) / (
        math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra)
    ) + 1.0 / (1 - math.exp(alpha * (1 - ra)))
    return bedroc


def full_metrics(scores, labels):
    """全指標を一括算出（AUC・EF@0.5/1/2/5%・BEDROC20）。lower=active-like。"""
    return {
        "AUC": roc_auc(scores, labels),
        "EF0.5": enrichment_factor(scores, labels, 0.005),
        "EF1": enrichment_factor(scores, labels, 0.01),
        "EF2": enrichment_factor(scores, labels, 0.02),
        "EF5": enrichment_factor(scores, labels, 0.05),
        "BEDROC20": bedroc(scores, labels, 20.0),
    }


if __name__ == "__main__":
    # 自己検証: 完全分離(active が全て最良)→AUC=1,BEDROC≈1。ランダム→AUC≈0.5。
    import random
    n, na = 1000, 50
    perfect_s = list(range(n)); perfect_l = [1] * na + [0] * (n - na)
    print("perfect: AUC=%.3f EF1=%.1f EF5=%.1f BEDROC=%.3f" % (
        roc_auc(perfect_s, perfect_l), enrichment_factor(perfect_s, perfect_l, 0.01),
        enrichment_factor(perfect_s, perfect_l, 0.05), bedroc(perfect_s, perfect_l)))
    random.seed(0)
    rs = [random.random() for _ in range(n)]; rl = [1] * na + [0] * (n - na); random.shuffle(rl)
    print("random : AUC=%.3f EF1=%.1f EF5=%.1f BEDROC=%.3f" % (
        roc_auc(rs, rl), enrichment_factor(rs, rl, 0.01),
        enrichment_factor(rs, rl, 0.05), bedroc(rs, rl)))
