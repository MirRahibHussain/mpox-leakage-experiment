"""Statistics for Section VI: protocol groups in the random sample (strata 1-2), the IEEE Xplore
open-access stratum (stratum 3) and all coded reports; abstract-level comparison of IEEE Xplore with
the other databases.  Run from this folder:  python audit_stats.py
Supersedes trend_test.py (same numbers for the random sample)."""
import csv, collections, numpy as np
from scipy import stats

rows = [r for r in csv.DictReader(open("coding_sheet.csv")) if r["full text coded"] == "yes"]

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def jt(groups):  # Jonckheere-Terpstra statistic, groups ordered low -> high
    return sum((y > x) + 0.5 * (y == x) for i in range(len(groups)) for j in range(i + 1, len(groups))
               for x in groups[i] for y in groups[j])

def jt_p(C, B, A, R=100_000):
    obs = jt([C, B, A]); pool = np.array(C + B + A); rng = np.random.default_rng(2026)
    return sum(jt([p[:len(C)], p[len(C):len(C) + len(B)], p[len(C) + len(B):]]) >= obs
               for p in (rng.permutation(pool) for _ in range(R))) / R

def report(sel, label, R=100_000):
    g = collections.Counter(r["group"] for r in sel); n = len(sel)
    print(f"\n{label}: n = {n}")
    for k in "ABCU":
        lo, hi = wilson(g[k], n); print(f"  {k}: {g[k]:2d}/{n} = {100 * g[k] / n:4.1f}%  (95% CI {lo:.0f}-{hi:.0f}%)")
    acc = {k: [float(r["headline accuracy (%)"]) for r in sel if r["group"] == k and r["headline accuracy (%)"]] for k in "ABCU"}
    print("  median headline accuracy:", {k: round(float(np.median(v)), 2) for k, v in acc.items() if v})
    if len(acc["C"]) > 1:
        print(f"  Jonckheere-Terpstra C<B<A, one-sided permutation p = {jt_p(acc['C'], acc['B'], acc['A'], R):.4f}")
        print(f"  Mann-Whitney A>C p = {stats.mannwhitneyu(acc['A'], acc['C'], alternative='greater').pvalue:.4f}; "
              f"A>B p = {stats.mannwhitneyu(acc['A'], acc['B'], alternative='greater').pvalue:.4f}")

sample = [r for r in rows if not r["stratum"].startswith("stratum 3")]
ieee = [r for r in rows if r["stratum"].startswith("stratum 3")]
report(sample, "Random sample (PubMed, Europe PMC, arXiv)")
report(ieee, "IEEE Xplore open-access stratum (census of IEEE Access)")
report(rows, "All coded reports")

ab = list(csv.DictReader(open("abstract_level_accuracy.csv")))
def summ(sel, label):
    a = [float(r["headline accuracy from abstract (%)"]) for r in sel if r["headline accuracy from abstract (%)"]]
    print(f"{label}: {len(sel)} records, {len(a)} with an accuracy in the abstract, median {np.median(a):.2f}%, "
          f">= 99%: {np.mean([v >= 99 for v in a]) * 100:.0f}%")
    return a
print("\nAbstract-level comparison (first accuracy figure stated in each abstract, extracted automatically):")
x = summ([r for r in ab if r["source"] == "IEEE Xplore"], "  IEEE Xplore")
y = summ([r for r in ab if r["source"] != "IEEE Xplore"], "  PubMed, arXiv, Europe PMC")
print(f"  Mann-Whitney two-sided p = {stats.mannwhitneyu(x, y).pvalue:.2f}")
