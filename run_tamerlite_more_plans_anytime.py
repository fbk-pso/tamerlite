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
from pathlib import Path
import csv

os.environ.setdefault("DISABLE_RUSTAMER", "1")
env = get_environment()
env.credits_stream = None
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")

# get_environment().credits_stream = None  # suppress credits header

def solve_and_summarize(
    problem_file, heuristic=None, timeout=30, max_len=None
):
    if problem_file.endswith(".anml"):
        problem = ANMLReader().parse_problem(str(problem_file))
    elif problem_file.endswith(".pddl"):
        problem = PDDLReader().parse_problem(str(problem_file))
    else:
        raise ValueError("Unsupported problem file format")

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
    parser.add_argument('--problem', required=True, type=Path, help='Path to database of valid plans')
    parser.add_argument('--heuristic', required=True, type=str, help='Heuristic')
    parser.add_argument('--timeout', required=True, type=float, help='Timeout')
    parser.add_argument('--output', required=True, type= Path, help="output csv file")
    parser.add_argument('--max_len', type=str, default=None, help='Max length macros')


    args, _ = parser.parse_known_args()

    if args.max_len is None:
        max_len = None
    elif args.max_len == "inf":
        max_len = float("inf")
    else:
        max_len = float(args.max_len)

    plans = solve_and_summarize(args.problem, heuristic=args.heuristic, timeout=args.timeout, max_len=max_len)

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

