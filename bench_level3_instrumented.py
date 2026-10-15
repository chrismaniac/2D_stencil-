
#!/usr/bin/env python3
"""
SIT315 M4.T1D - Level 3 Instrumented MPI Benchmark

Compares blocking MPI halo communication with non-blocking
communication overlapped with interior stencil computation.

Records:
- Execution time
- Communication fraction
- MPI request posting time
- Interior computation time
- MPI waiting time
- Border computation time
- Checksum verification

Each configuration runs seven times, with alternating
execution order to reduce systematic order bias.
"""

import csv
import statistics
import subprocess
from pathlib import Path

# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

SIZES = [1024, 2048, 4096]
PROCESSES = [2, 4]
ITERATIONS = 200
REPEATS = 7

EXECUTABLE = "./stencil_level3"

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULTS / "level3_instrumented.csv"

# --------------------------------------------------
# Execute MPI program
# --------------------------------------------------

def run_mpi(size, processes, mode):

    command = [
        "mpirun",
        "-np",
        str(processes),
        EXECUTABLE,
        str(size),
        str(ITERATIONS)
    ]

    if mode == "overlap":
        command.append("--overlap")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MPI execution failed:\n{result.stderr}"
        )

    for line in result.stdout.splitlines():

        if line.startswith("STATS"):

            return {
                key: value
                for key, value in (
                    part.split("=", 1)
                    for part in line.split()[1:]
                )
            }

    raise RuntimeError(
        "No STATS output found:\n" + result.stdout
    )

# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    rows = []

    print("=" * 60)
    print("SIT315 LEVEL 3 - INSTRUMENTED MPI BENCHMARK")
    print("=" * 60)

    print(f"Grid sizes: {SIZES}")
    print(f"MPI processes: {PROCESSES}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Repetitions: {REPEATS}")

    for size in SIZES:

        # Sequential reference for correctness
        seq_command = [
            "./stencil_seq",
            str(size),
            str(ITERATIONS)
        ]

        seq_result = subprocess.run(
            seq_command,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )

        seq_line = next(
            line for line in seq_result.stdout.splitlines()
            if line.startswith("STATS")
        )

        seq_stats = dict(
            part.split("=", 1)
            for part in seq_line.split()[1:]
        )

        reference_checksum = seq_stats["checksum"]

        for processes in PROCESSES:

            print(
                f"\nTesting grid={size}x{size}, "
                f"np={processes}",
                flush=True
            )

            for repeat in range(REPEATS):

                # Alternate order between consecutive repetitions
                if repeat % 2 == 0:
                    modes = ["blocking", "overlap"]
                else:
                    modes = ["overlap", "blocking"]

                for mode in modes:

                    result = run_mpi(
                        size,
                        processes,
                        mode
                    )

                    # Verify against sequential reference
                    if result["checksum"] != reference_checksum:

                        raise RuntimeError(
                            f"CHECKSUM MISMATCH: "
                            f"N={size}, np={processes}, "
                            f"mode={mode}"
                        )

                    row = {
                        "N": size,
                        "np": processes,
                        "repeat": repeat + 1,
                        "mode": mode,

                        "time": float(result["time"]),

                        "comm_frac": float(
                            result["comm_frac"]
                        ),

                        "post_avg": float(
                            result["post_avg"]
                        ),

                        "interior_avg": float(
                            result["interior_avg"]
                        ),

                        "wait_avg": float(
                            result["wait_avg"]
                        ),

                        "border_avg": float(
                            result["border_avg"]
                        ),

                        "checksum": result["checksum"]
                    }

                    rows.append(row)

                    print(
                        f"Run {repeat + 1}: "
                        f"{mode:<8} "
                        f"time={row['time']:.4f}s "
                        f"comm={row['comm_frac']:.3f} "
                        f"checksum=OK",
                        flush=True
                    )

    # --------------------------------------------------
    # Save detailed CSV results
    # --------------------------------------------------

    with OUTPUT.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Final performance summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 60)

    for size in SIZES:

        for processes in PROCESSES:

            samples = {}

            for mode in ("blocking", "overlap"):

                samples[mode] = [
                    row
                    for row in rows
                    if row["N"] == size
                    and row["np"] == processes
                    and row["mode"] == mode
                ]

            blocking_time = statistics.median(
                row["time"]
                for row in samples["blocking"]
            )

            overlap_time = statistics.median(
                row["time"]
                for row in samples["overlap"]
            )

            improvement = (
                (blocking_time - overlap_time)
                / blocking_time
                * 100
            )

            # Phase measurements only apply to overlap mode
            overlap_post = statistics.median(
                row["post_avg"]
                for row in samples["overlap"]
            )

            overlap_interior = statistics.median(
                row["interior_avg"]
                for row in samples["overlap"]
            )

            overlap_wait = statistics.median(
                row["wait_avg"]
                for row in samples["overlap"]
            )

            overlap_border = statistics.median(
                row["border_avg"]
                for row in samples["overlap"]
            )

            print(
                f"\nN={size}, np={processes}"
            )

            print(
                f"  Blocking median: {blocking_time:.4f}s"
            )

            print(
                f"  Overlap median:  {overlap_time:.4f}s"
            )

            print(
                f"  Improvement:     {improvement:+.2f}%"
            )

            print("  Overlap phase times (mean per rank):")

            print(
                f"    Posting:  {overlap_post:.6f}s"
            )

            print(
                f"    Interior: {overlap_interior:.6f}s"
            )

            print(
                f"    Waiting:  {overlap_wait:.6f}s"
            )

            print(
                f"    Border:   {overlap_border:.6f}s"
            )

    print("\n" + "=" * 60)
    print(f"Saved detailed results: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
