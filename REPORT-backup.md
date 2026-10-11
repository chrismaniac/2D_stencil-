# M4.T1D Project – Distributed 2D Heat Diffusion (MPI domain decomposition)
*Claimed tier: Level 2 (extras below are Level 3 material – delete the ones you do not demonstrate).*

## 1. Scope
Simulate heat diffusion on an N×N plate (Jacobi 5-point stencil, fixed 0 boundary, two initial hot discs) for a fixed
number of sweeps, distributed across MPI ranks. Deliverables: sequential reference, MPI program, benchmark driver,
snapshot renderer for the demo video. Success criterion: MPI output is **bit-identical** to the sequential output
for every process count/decomposition (checked by an order-independent checksum), and runtime/speedup is measured on 3+ sizes.

## 2. Design
- **Decomposition:** `MPI_Dims_create` + `MPI_Cart_create` → R×C process grid; each rank owns an `ly × lx` block (uneven N handled, first `N mod R` rows get one extra).
- **Halo (ghost) cells:** each block is stored with a 1-cell ring. N/S halo rows are contiguous; W/E halo columns are strided → `MPI_Type_vector`. Global edges have neighbour `MPI_PROC_NULL`, so the fixed boundary needs no special code.
- **Exchange (baseline):** 4 × `MPI_Sendrecv` (rows, then columns) – deadlock-free by construction.
- **Why 2D beats 1D (theory to verify):** per sweep compute ∝ N²/P cells; halo volume ∝ N/√P for a 2D grid vs ∝ N for 1D strips (independent of P), so the communication/computation ratio grows slower with P in 2D.
- **Correctness:** the update expression is identical in seq/MPI and compiled with `-ffp-contract=off`, so results are bit-identical; checksum = wrapping sum of a hash of (cell index, bits), independent of summation order.
- **Level 3 extras (optional, already implemented):** `--overlap` (Isend/Irecv, interior computed while halos are in flight, border computed after `Waitall`); `stencil_hybrid` (OpenMP threads inside each rank, `MPI_THREAD_FUNNELED`); `--dims P 1` (1D strips) for the decomposition comparison; MPI-IO snapshots (`MPI_Type_create_subarray`).

## 3. What is new compared with the unit materials / research done
List honestly what you had to learn: Cartesian topologies (`MPI_Cart_create/shift/coords`), derived datatypes for strided columns, `MPI_PROC_NULL` for boundaries, non-blocking overlap pattern, subarray datatypes + collective MPI-IO, MPI thread levels, surface-to-volume analysis. Add the sources you actually read (MPI standard / lecture notes / books / tutorials).

## 4. Evaluation
**System (fill in from `./sysinfo.sh`):** CPU model ___, ___ cores / ___ threads, ___ GB RAM, OS ___, MPI ___ (`mpirun --version`), compiler ___.
**Method:** `python3 bench.py scaling --sizes <3 sizes> --iters 200 --np 1 2 4 8 --reps 3`; median of 3 runs (min/max reported); every MPI run checked against the sequential checksum; speedup = T_seq / T_mpi; efficiency = speedup / np. Choose sizes so runtimes clearly increase (e.g. each ≈ 4× the previous cell count) and np ≤ your physical cores.
**Results:** paste `results/scaling.md` here + one chart (speedup vs np per size).
**Analysis prompts (answer with YOUR numbers):**
1. Did speedup occur? At which size/np? Why is the smallest size slow or even < 1 (fixed MPI latency vs tiny compute)?
2. Does speedup rise with N for fixed np? (computation grows as N², halos as N.)
3. Where does efficiency drop? Use `comm_frac`. Memory-bandwidth saturation: stencil is memory-bound, so cores share bandwidth.
4. 2D vs 1D decomposition – does `comm_frac` differ as np grows? Matches the N/√P vs N argument?
5. (Level 3) Does `--overlap` help? It only helps if there is enough interior work and the MPI library makes progress during compute (many implementations progress only inside MPI calls).
6. (Level 3) Hybrid vs pure MPI at the same total core count.
7. Oversubscription (np > physical cores) – what happens and why.

## 5. Stress test (required)
`./stress.sh 40000 500 8 600` (adjust N so it fails or runs long **on your machine**). Record: machine specs, command, what happened (finished / out of memory / "waited 10 minutes, DNF"), and the reason. Memory estimate: `2·(N+2)²·8` bytes in total.

## 6. Limitations / future work
Fixed iteration count (no convergence test – would need `MPI_Allreduce` of the residual); fixed 5-point stencil; square domain; periodic boundaries unsupported; snapshots written by all ranks collectively (can dominate runtime if dumped often).
