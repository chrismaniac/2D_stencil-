#!/usr/bin/env python3
"""Benchmark driver for the stencil project. Every MPI run is checked against the sequential checksum.

  python3 bench.py scaling --sizes 1024 2048 4096 --iters 200 --np 1 2 4 8 --reps 3
  python3 bench.py level3  --size 4096 --iters 200 --np 8 --threads 2      (Level 3 extras)
Use --mpirun "mpirun --oversubscribe" on Open MPI if np > cores. Results -> results/*.md"""
import argparse, os, shlex, subprocess, sys

def stats(line): return dict(kv.split("=", 1) for kv in line.split()[1:])
def run(cmd, env=None, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    for l in (p.stdout + p.stderr).splitlines()[::-1]:
        if l.startswith("STATS"): return stats(l)
    raise RuntimeError(f"no STATS from {cmd}\n{p.stderr[-400:]}")
def median(cmd, reps, env=None):
    rs = sorted((run(cmd, env) for _ in range(reps)), key=lambda r: float(r["time"]))
    r = rs[len(rs) // 2]; r["_min"] = rs[0]["time"]; r["_max"] = rs[-1]["time"]; return r

def scaling(a):
    mr = shlex.split(a.mpirun); rows = []; os.makedirs("results", exist_ok=True)
    for N in a.sizes:
        seq = median(["./stencil_seq", str(N), str(a.iters)], a.reps); ts = float(seq["time"])
        rows.append((N, "sequential", 1, "-", ts, 1.0, 1.0, "-", seq["_min"], seq["_max"]))
        print(f"N={N} seq {ts:.3f}s", flush=True)
        for np_ in a.np:
            for label, extra in (("2D blocking", []), ("2D overlap", ["--overlap"]), ("1D blocking", ["--dims", str(np_), "1"])):
                r = median(mr + ["-np", str(np_), "./stencil_mpi", str(N), str(a.iters)] + extra, a.reps)
                if r["checksum"] != seq["checksum"]: sys.exit(f"CHECKSUM MISMATCH N={N} np={np_} {label}")
                t = float(r["time"]); sp = ts / t
                rows.append((N, label, np_, r["dims"], t, sp, sp / np_, r["comm_frac"], r["_min"], r["_max"]))
                print(f"N={N} np={np_} {label:12s} dims={r['dims']:>5s} {t:8.3f}s speedup={sp:5.2f} eff={sp/np_:4.2f} comm={r['comm_frac']} OK", flush=True)
    with open("results/scaling.md", "w") as o:
        o.write(f"Median of {a.reps} runs, {a.iters} sweeps. Checksums verified against sequential.\n\n")
        o.write("| N (N x N) | version | np | dims | time s (median) | min | max | speedup | efficiency | comm fraction |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for N, lab, np_, d, t, sp, ef, cf, mn, mx in rows:
            o.write(f"| {N} | {lab} | {np_} | {d} | {t:.3f} | {mn} | {mx} | {sp:.2f} | {ef:.2f} | {cf} |\n")
    print("wrote results/scaling.md")

def level3(a):
    mr = shlex.split(a.mpirun); N = a.size; P = a.np; out = []
    seq = run(["./stencil_seq", str(N), str(a.iters)]); ref = seq["checksum"]
    print(f"seq {seq['time']}s")
    hdr = "| experiment | np x threads | dims | time s | comm fraction | check |\n|---|---|---|---|---|---|"; out.append(hdr); print(hdr)
    def row(tag, cmd, env=None):
        r = median(cmd, a.reps, env); ok = "OK" if r["checksum"] == ref else "MISMATCH"
        s = f"| {tag} | {r['np']} x {r['threads']} | {r['dims']} | {float(r['time']):.3f} | {r['comm_frac']} | {ok} |"; print(s, flush=True); out.append(s)
    b = mr + ["-np", str(P), "./stencil_mpi", str(N), str(a.iters)]
    row("pure MPI, 2D, blocking", b)
    row("pure MPI, 2D, overlap", b + ["--overlap"])
    row("pure MPI, 1D strips, blocking", b + ["--dims", str(P), "1"])
    row("pure MPI, 1D strips, overlap", b + ["--dims", str(P), "1", "--overlap"])
    if P % a.threads == 0:
        h = mr + ["-np", str(P // a.threads), "./stencil_hybrid", str(N), str(a.iters)]; env = {"OMP_NUM_THREADS": str(a.threads)}
        row(f"hybrid MPI+OpenMP, blocking", h, env); row(f"hybrid MPI+OpenMP, overlap", h + ["--overlap"], env)
    open("results/level3.md", "w").write("\n".join(out) + "\n"); print("wrote results/level3.md")

ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="mode", required=True)
for n in ("scaling", "level3"):
    p = sp.add_parser(n); p.add_argument("--mpirun", default="mpirun"); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--iters", type=int, default=200)
    if n == "scaling": p.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096]); p.add_argument("--np", type=int, nargs="+", default=[1, 2, 4, 8])
    else: p.add_argument("--size", type=int, default=4096); p.add_argument("--np", type=int, default=8); p.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); {"scaling": scaling, "level3": level3}[a.mode](a)
