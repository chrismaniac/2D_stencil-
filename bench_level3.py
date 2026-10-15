
#!/usr/bin/env python3
"""Focused Level 3 MPI blocking-versus-overlap benchmark."""

import csv
import os
import statistics
import subprocess
from pathlib import Path

SIZES = [1024, 2048, 4096]
PROCESSES = [2, 4]
ITERATIONS = 200
REPEATS = 7

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def run_mpi(size, processes, mode):
    cmd = [
        "mpirun", "-np", str(processes),
        "./stencil_mpi", str(size), str(ITERATIONS)
    ]

    if mode == "overlap":
        cmd.append("--overlap")

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    for line in result.stdout.splitlines():
        if line.startswith("STATS"):
            return {
                k: v
                for k, v in (
                    part.split("=", 1)
                    for part in line.split()[1:]
                )
            }

    raise RuntimeError("Missing STATS output")


def main():
    rows = []

    for size in SIZES:
        for processes in PROCESSES:
            print(f"\nTesting N={size}, np={processes}", flush=True)

            reference = None

            for repeat in range(REPEATS):
                # Alternate the execution order.
                modes = (
                    ["blocking", "overlap"]
                    if repeat % 2 == 0
                    else ["overlap", "blocking"]
                )

                for mode in modes:
                    result = run_mpi(size, processes, mode)

                    if reference is None:
                        reference = result["checksum"]

                    if result["checksum"] != reference:
                        raise RuntimeError(
                            "Checksum mismatch detected!"
                        )

                    row = {
                        "N": size,
                        "np": processes,
                        "repeat": repeat + 1,
                        "mode": mode,
                        "time": float(result["time"]),
                        "comm_frac": float(result["comm_frac"]),
                        "checksum": result["checksum"]
                    }

                    rows.append(row)

                    print(
                        f"Run {repeat + 1}: {mode:8s} "
                        f"{row['time']:.4f}s OK",
                        flush=True
                    )

    output = RESULTS / "level3_detailed.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nFINAL SUMMARY")

    for size in SIZES:
        for processes in PROCESSES:
            times = {}

            for mode in ("blocking", "overlap"):
                times[mode] = [
                    r["time"] for r in rows
                    if r["N"] == size
                    and r["np"] == processes
                    and r["mode"] == mode
                ]

            blocking = statistics.median(times["blocking"])
            overlap = statistics.median(times["overlap"])

            improvement = (
                (blocking - overlap) / blocking * 100
            )

            print(
                f"N={size} np={processes}: "
                f"blocking={blocking:.4f}s, "
                f"overlap={overlap:.4f}s, "
                f"improvement={improvement:+.2f}%"
            )

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
