"""Trend test for Section VI / Fig. 5a.
Headline accuracies from the rebuilt coding sheet (Supplementary_search_log_and_coding_sheet.xlsx). Run:  python jt_test.py
"""
import numpy as np
from scipy import stats

A = [87.13, 93.03, 93.29, 94.0, 94.56, 95.0, 96.52, 97.63, 97.8, 98.3, 98.4, 98.78]  # augmented copies reach test set
B = [82.47, 84.8, 93.0, 93.15, 94.2, 95.45, 96.0, 96.51, 96.7, 97.0, 97.5, 98.0, 98.81]  # unique images, no patient grouping
C = [82.96, 83.59, 83.89, 87.3, 90.38, 91.11, 93.71, 97.8]  # patient-grouped

def jt(groups):  # Jonckheere-Terpstra statistic, groups ordered low -> high
    return sum((y > x) + 0.5 * (y == x)
               for i in range(len(groups)) for j in range(i + 1, len(groups))
               for x in groups[i] for y in groups[j])

obs = jt([C, B, A])
pool = np.array(C + B + A); rng = np.random.default_rng(2026); R = 100_000
hits = sum(jt([p[:len(C)], p[len(C):len(C) + len(B)], p[len(C) + len(B):]]) >= obs
           for p in (rng.permutation(pool) for _ in range(R)))
print("medians A/B/C:", np.median(A), np.median(B), np.median(C))
print(f"Jonckheere-Terpstra (C<B<A), one-sided permutation p = {hits / R:.4f}")
print("A > C:", stats.mannwhitneyu(A, C, alternative="greater").pvalue)
print("A > B:", stats.mannwhitneyu(A, B, alternative="greater").pvalue)
print("Kruskal-Wallis:", stats.kruskal(A, B, C).pvalue)
