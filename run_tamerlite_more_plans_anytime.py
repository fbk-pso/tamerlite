#!/usr/bin/env python3

# HOW TO RUN: python3 run_tamerlite_more_plans_anytime.py --problem <path_to_problem> --heuristic hff --output <path_to_output> --timeout 20 --max_len <'inf' or upper bound>

# early_termination must be false

import os

from unified_planning.shortcuts import *
from unified_planning.io import ANMLReader, PDDLReader
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import PlanKind

from tamerlite import HeuristicParams, MultiqueueParams, SearchParams
from tamerlite.engine import TamerLite

import argparse
import csv

os.environ.setdefault("DISABLE_RUSTAMER", "1")
env = get_environment()
env.credits_stream = None
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")

# get_environment().credits_stream = None  # suppress credits header

def solve_and_summarize(
    problem, heuristic=None, timeout=30, max_len=None
):

    params = SearchParams(
        search="wastar",
        heuristic=heuristic,
        weight=0.8,
        max_len=max_len,
    )   

    with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
        res = planner.solve(problem, timeout=timeout)
        print(res)
        print(f"Expanded states: {res.metrics["expanded_states"]}")

    plans = res.metrics.get("plans", [])

    return plans


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--problem', required=True, type=str, help='Path to database of valid plans')
    parser.add_argument('--domain', default=None, type=str, help='Path to domain file')
    parser.add_argument('--heuristic', required=True, type=str, help='Heuristic')
    parser.add_argument('--timeout', required=True, type=float, help='Timeout')
    parser.add_argument('--output', required=True, type= str, help="output csv file")
    parser.add_argument('--max_len', type=str, default=None, help='Max length macros')


    args, _ = parser.parse_known_args()

    if args.max_len is None:
        max_len = None
    elif args.max_len == "inf":
        max_len = float("inf")
    else:
        max_len = float(args.max_len)

    if args.problem.endswith(".anml"):
        problem = ANMLReader().parse_problem(args.problem)
    elif args.problem.endswith(".pddl"):
        assert args.domain is not None, "Domain file must be provided for PDDL problems"
        problem = PDDLReader().parse_problem(args.domain, args.problem)
    else:
        raise ValueError("Unsupported problem file format")

    plans = solve_and_summarize(problem, heuristic=args.heuristic, timeout=args.timeout, max_len=max_len)

    print(f"Found {len(plans)} plans")

    for plan in plans:
        print(plan)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        for actions_trace in plans:
            trace_str = "|".join(actions_trace)
            writer.writerow([trace_str])


if __name__ == '__main__':
    main()

