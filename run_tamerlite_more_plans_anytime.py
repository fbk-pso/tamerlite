#!/usr/bin/env python3
# Enumerate all plans with at most max_len events (inf: length of the first plan found).
# Usage: python3 run_tamerlite_more_plans_anytime.py --problem P.anml --heuristic hadd \
#            --timeout 600 --max_len inf --output plans.csv [--symmetry_breaking]

import argparse
import csv
import json

from unified_planning.io import ANMLReader, PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment

from tamerlite import SearchParams

env = get_environment()
env.credits_stream = None
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--heuristic", required=True)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_len", default="inf")
    parser.add_argument("--weight", default=0.8, type=float)
    parser.add_argument("--symmetry_breaking", action="store_true")
    args = parser.parse_args()

    if args.problem.endswith(".anml"):
        problem = ANMLReader().parse_problem(args.problem)
    else:
        assert args.domain is not None, "PDDL problems need --domain"
        problem = PDDLReader().parse_problem(args.domain, args.problem)

    params = SearchParams(
        search="wastar",
        heuristic=args.heuristic,
        weight=args.weight,
        max_len=float(args.max_len),
        incomplete_memory_bounded_search=True,
        symmetry_breaking=args.symmetry_breaking,
    )
    with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
        res = planner.solve(problem, timeout=args.timeout)

    plans = res.metrics.get("plans", [])
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for plan in plans:
            writer.writerow(["|".join(plan)])

    lengths = [len(p) for p in plans]
    summary = {
        "status": res.status.name,
        "expanded_states": res.metrics.get("expanded_states"),
        "n_plans": len(plans),
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
