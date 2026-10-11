# 2D Heat Diffusion – MPI (SIT315 M4.T1D)
Files: `stencil_seq.c`, `stencil_mpi.c` (builds both the MPI and the MPI+OpenMP binary), `stencil_common.h`, `bench.py`, `render.py`, `stress.sh`, `sysinfo.sh`, `Makefile`, `REPORT.md` (project document template).

## Compile (exact commands; `make` runs all three)
    gcc   -O2 -ffp-contract=off -o stencil_seq stencil_seq.c -lm
    mpicc -O2 -ffp-contract=off -o stencil_mpi stencil_mpi.c -lm
    mpicc -O2 -ffp-contract=off -fopenmp -o stencil_hybrid stencil_mpi.c -lm

## Run
    ./stencil_seq 1024 200
    mpirun -np 4 ./stencil_mpi 1024 200                       # 2D grid chosen automatically (2x2)
    mpirun -np 4 ./stencil_mpi 1024 200 --overlap             # non-blocking halo exchange, overlap with compute
    mpirun -np 4 ./stencil_mpi 1024 200 --dims 4 1            # 1D strips (for the comparison)
    OMP_NUM_THREADS=2 mpirun -np 4 ./stencil_hybrid 1024 200  # 4 ranks x 2 threads
    mpirun -np 4 ./stencil_mpi 400 600 --dump data/snap 100 && python3 render.py data/snap 400   # images for the video
Output line `STATS ...` includes time, comm fraction and a checksum – the checksum must equal the sequential one.
Open MPI with np > cores: `mpirun --oversubscribe -np 8 ...`.

## Tonight's plan (deadline 12am)
1. (15 min) `make`, run the commands above, check seq vs MPI checksums, screenshot.
2. (30–60 min) Run `bench.py scaling` with sizes chosen for your machine → `results/scaling.md`. Start small and time one run first; pick 3 sizes where the sequential time is roughly 1 s, 4 s, 16 s.
3. (10 min) `./sysinfo.sh` + `./stress.sh ...` for the required stress test.
4. (30 min) Fill `REPORT.md` (sections 3–5 need your own numbers/words).
5. (45+ min) Record the 10+ minute video (outline below). Submit a working Level 2 first, THEN try `bench.py level3` if time remains.

## Video outline (10+ min)
Problem and scope (1) · decomposition + halo diagram on screen (2) · code walk-through of Cart_create, halo exchange, Type_vector, overlap (3) · live run + checksum match (1) · heat-map snapshots (1) · benchmark table and speedup analysis (2) · stress test and machine specs (1) · what was new/researched and limitations (1).

## Likely interview questions
Why is the checksum order-independent? · What happens at global edges (PROC_NULL)? · Why does the column need a derived datatype? · Why Sendrecv rather than Send then Recv? · Surface-to-volume: 1D vs 2D? · Why may overlap not help? · Why is speedup < 1 on small grids?
