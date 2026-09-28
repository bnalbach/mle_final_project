"""
Compare two evaluate.py result JSONs. Prints metrics and delta.

    python compare.py results/before.json results/after.json
"""
import json
import sys

# metrics worth showing (JSON aggregate keys), and whether "up is good"
METRICS = [
    ("outright_win", True), ("rank", False), ("score", True),
    ("kills", True), ("suicides", False), ("survived", True),
    ("steps_alive", True), ("stuck", False), ("coin_share", True),
]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python compare.py <before.json> <after.json>")
    a, b = load(sys.argv[1]), load(sys.argv[2])
    for name in a:
        if name not in b:
            continue
        ga, gb = a[name]["aggregate"], b[name]["aggregate"]
        print(f"\n=== {name} ===")
        print(f"    {'metric':<14}{'before':>10}{'after':>10}{'delta':>10}")
        for key, up_good in METRICS:
            if key not in ga or key not in gb:
                continue
            va, vb = ga[key], gb[key]
            d = vb - va
            better = (d > 0) == up_good if abs(d) > 1e-9 else None
            mark = "" if better is None else ("  +" if better else "  -")
            print(f"    {key:<14}{va:>10.3f}{vb:>10.3f}{d:>+10.3f}{mark}")


if __name__ == "__main__":
    main()
