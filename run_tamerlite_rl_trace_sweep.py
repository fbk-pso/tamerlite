#!/usr/bin/env python3

import argparse
import csv
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNNER = Path(__file__).resolve().parent / "run_tamerlite_more_plans_anytime.py"
DEFAULT_TRACES_ROOT = Path("/storage/PSO/kgrover/new_pruning")
DEFAULT_EXTRA_TRACES_ROOT = Path("/storage/PSO/kgrover/extra_traces")
DEFAULT_DOMAINS = ("replenish", "kitting", "childsnack", "blocksworld")
DEFAULT_HEURISTICS = ("hadd", "hff")

DOMAIN_CONFIGS = {
    "replenish": {
        "traces_group": "temporal",
        "output_group": "temporal",
        "benchmark_root": REPO_ROOT / "experiments" / "custom_cluster" / "temporal" / "replenish",
    },
    "kitting": {
        "traces_group": "temporal",
        "output_group": "temporal",
        "benchmark_root": REPO_ROOT / "experiments" / "custom_cluster" / "temporal" / "kitting",
    },
    "childsnack": {
        "traces_group": "classic_small",
        "output_group": "classic_small",
        "benchmark_root": REPO_ROOT / "experiments" / "custom_cluster" / "classic" / "childsnack",
    },
    "blocksworld": {
        "traces_group": "classic_small",
        "output_group": "classic_small",
        "benchmark_root": REPO_ROOT / "experiments" / "custom_cluster" / "classic" / "blocksworld",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Path to one benchmark directory containing traces.csv, e.g. /storage/PSO/.../<group>/<benchmark>.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        choices=sorted(DOMAIN_CONFIGS),
        help="Benchmarks to run when --benchmark-dir is not provided.",
    )
    parser.add_argument(
        "--heuristics",
        nargs="+",
        default=list(DEFAULT_HEURISTICS),
        choices=("hadd", "hff"),
        help="Heuristics to run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Planner timeout in seconds.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the TamerLite runner.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="Path to tamerlite/run_tamerlite_more_plans_anytime.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for plans, logs, and manifests. Defaults to /storage/PSO/kgrover/extra_traces/<group>/<benchmark>/.",
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=DEFAULT_TRACES_ROOT,
        help="Root directory containing <group>/<benchmark>/traces.csv when --benchmark-dir is not provided.",
    )
    parser.add_argument(
        "--extra-traces-root",
        type=Path,
        default=DEFAULT_EXTRA_TRACES_ROOT,
        help="Base directory used to build the default output path.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose output CSV already exists.",
    )
    parser.add_argument(
        "--emit-slurm-jobs",
        action="store_true",
        help="Generate one Slurm job script per instance instead of executing locally.",
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=None,
        help="Directory where Slurm job scripts will be written. Defaults to <output-root>/slurm.",
    )
    parser.add_argument(
        "--slurm-time",
        default="00:10:00",
        help="Slurm wall clock limit for each generated job.",
    )
    parser.add_argument(
        "--slurm-mem",
        default="4G",
        help="Slurm memory request for each generated job.",
    )
    parser.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=1,
        help="Slurm CPU request for each generated job.",
    )
    parser.add_argument(
        "--slurm-partition",
        default=None,
        help="Optional Slurm partition for generated jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def solved_lengths_by_problem(traces_path: Path) -> dict[int, int]:
    lengths: dict[int, list[int]] = defaultdict(list)
    with traces_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {key.strip(): value.strip() for key, value in row.items()}
            if normalized["solved"] != "True":
                continue
            trace = normalized["trace"]
            if not trace:
                continue
            problem_id = int(normalized["problem_id"])
            lengths[problem_id].append(len(trace.split("|")))
    return {problem_id: min(values) for problem_id, values in lengths.items()}


def list_problem_files(instances_dir: Path) -> list[tuple[int, Path]]:
    pairs = []
    for path in instances_dir.glob("problem_*.anml"):
        suffix = path.stem.removeprefix("problem_")
        pairs.append((int(suffix), path))
    return sorted(pairs, key=lambda item: item[0])


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_command(
    python_exe: str,
    runner_path: Path,
    problem_path: Path,
    heuristic: str,
    timeout: float,
    output_path: Path,
    max_len: int | None,
) -> list[str]:
    return [
        python_exe,
        str(runner_path),
        "--problem",
        str(problem_path),
        "--heuristic",
        heuristic,
        "--timeout",
        str(timeout),
        "--output",
        str(output_path),
        "--max_len",
        "inf" if max_len is None else str(max_len),
    ]


def write_slurm_job(
    job_path: Path,
    command: list[str],
    log_path: Path,
    job_name: str,
    slurm_time: str,
    slurm_mem: str,
    slurm_cpus_per_task: int,
    slurm_partition: str | None,
) -> None:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --time={slurm_time}",
        f"#SBATCH --mem={slurm_mem}",
        f"#SBATCH --cpus-per-task={slurm_cpus_per_task}",
    ]
    if slurm_partition:
        lines.append(f"#SBATCH --partition={slurm_partition}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            shell_join(command),
            "",
        ]
    )
    job_path.write_text("\n".join(lines), encoding="utf-8")
    job_path.chmod(0o755)


def resolve_domain_specs(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    if args.benchmark_dir is not None:
        benchmark_dir = args.benchmark_dir.resolve()
        domain_name = benchmark_dir.name
        if domain_name not in DOMAIN_CONFIGS:
            raise SystemExit(f"Unsupported benchmark name inferred from --benchmark-dir: {domain_name}")
        config = DOMAIN_CONFIGS[domain_name]
        return [
            {
                "domain": domain_name,
                "traces_path": benchmark_dir / "training" / "traces.csv",
                "instances_dir": benchmark_dir / "training" / "instances_sets" / "set_1" / "testing_set",
                "output_group": benchmark_dir.parent.name,
            }
        ]

    traces_root = args.traces_root.resolve()
    specs = []
    for domain_name in args.domains:
        config = DOMAIN_CONFIGS[domain_name]
        specs.append(
            {
                "domain": domain_name,
                "traces_path": traces_root / config["traces_group"] / domain_name / "traces.csv",
                "instances_dir": config["benchmark_root"] / "training" / "instances_sets" / "set_1" / "testing_set",
                "output_group": config["output_group"],
            }
        )
    return specs


def resolve_output_root(args: argparse.Namespace, output_group: str, domain_name: str) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    return (args.extra_traces_root.resolve() / output_group / domain_name).resolve()


def main() -> int:
    args = parse_args()
    runner_path = args.runner.resolve()
    ensure_exists(runner_path, "runner script")

    for spec in resolve_domain_specs(args):
        domain_name = str(spec["domain"])
        traces_path = Path(spec["traces_path"])
        instances_dir = Path(spec["instances_dir"])
        output_group = str(spec["output_group"])
        output_root = resolve_output_root(args, output_group, domain_name)
        slurm_dir = args.slurm_dir.resolve() if args.slurm_dir is not None else (output_root / "slurm").resolve()

        ensure_exists(traces_path, f"{domain_name} traces.csv")
        ensure_exists(instances_dir, f"{domain_name} instances directory")

        optimal_lengths = solved_lengths_by_problem(traces_path)
        problem_files = list_problem_files(instances_dir)
        if not problem_files:
            raise SystemExit(f"No problem_*.anml files found in {instances_dir}")

        manifest_rows: list[dict[str, str]] = []
        submit_commands: list[str] = []

        for heuristic in args.heuristics:
            heuristic_output_dir = output_root / heuristic
            heuristic_output_dir.mkdir(parents=True, exist_ok=True)

            for problem_id, problem_path in problem_files:
                output_path = heuristic_output_dir / f"problem_{problem_id}.csv"
                log_path = heuristic_output_dir / f"problem_{problem_id}.log"
                max_len = optimal_lengths.get(problem_id)
                max_len_source = "rl_trace_optimal" if max_len is not None else "fallback_inf"
                job_path = slurm_dir / heuristic / f"problem_{problem_id}.sbatch"
                job_name = f"tamer_{domain_name}_{heuristic}_{problem_id}"

                command = build_command(
                    python_exe=args.python,
                    runner_path=runner_path,
                    problem_path=problem_path,
                    heuristic=heuristic,
                    timeout=args.timeout,
                    output_path=output_path,
                    max_len=max_len,
                )

                manifest_rows.append(
                    {
                        "group": output_group,
                        "domain": domain_name,
                        "heuristic": heuristic,
                        "problem_id": str(problem_id),
                        "problem": str(problem_path),
                        "traces": str(traces_path),
                        "output": str(output_path),
                        "log": str(log_path),
                        "max_len": "inf" if max_len is None else str(max_len),
                        "max_len_source": max_len_source,
                        "slurm_job": str(job_path) if args.emit_slurm_jobs else "",
                    }
                )

                if args.skip_existing and output_path.exists():
                    print(f"SKIP existing {output_path}")
                    continue

                if args.emit_slurm_jobs:
                    job_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    write_slurm_job(
                        job_path=job_path,
                        command=command,
                        log_path=log_path,
                        job_name=job_name,
                        slurm_time=args.slurm_time,
                        slurm_mem=args.slurm_mem,
                        slurm_cpus_per_task=args.slurm_cpus_per_task,
                        slurm_partition=args.slurm_partition,
                    )
                    submit_commands.append(f"sbatch {shlex.quote(str(job_path))}")
                    print(f"WROTE {job_path}")
                    continue

                print(shell_join(command))
                if args.dry_run:
                    continue

                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as log_handle:
                    result = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if result.returncode != 0:
                    print(f"FAILED {domain_name} {heuristic} problem_{problem_id}: see {log_path}", file=sys.stderr)
                    return result.returncode

        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "group",
                    "domain",
                    "heuristic",
                    "problem_id",
                    "problem",
                    "traces",
                    "output",
                    "log",
                    "max_len",
                    "max_len_source",
                    "slurm_job",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        if args.emit_slurm_jobs:
            slurm_dir.mkdir(parents=True, exist_ok=True)
            submit_all_path = slurm_dir / "submit_all.sh"
            submit_all_path.write_text(
                "#!/bin/bash\nset -euo pipefail\n" + "\n".join(submit_commands) + ("\n" if submit_commands else ""),
                encoding="utf-8",
            )
            submit_all_path.chmod(0o755)
            print(f"Wrote submit script: {submit_all_path}")

        fallback_count = sum(1 for row in manifest_rows if row["max_len_source"] == "fallback_inf")
        print(f"Wrote manifest: {manifest_path}")
        print(f"Output root: {output_root}")
        print(f"Runs planned: {len(manifest_rows)}")
        print(f"Fallback max_len=inf runs: {fallback_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
