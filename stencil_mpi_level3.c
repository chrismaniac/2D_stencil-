/* stencil_mpi.c - 2D heat diffusion (Jacobi 5-point) with MPI domain decomposition.
 *
 * Build : mpicc -O2 -ffp-contract=off -o stencil_mpi stencil_mpi.c -lm            (MPI only)
 *         mpicc -O2 -ffp-contract=off -fopenmp -o stencil_hybrid stencil_mpi.c -lm (MPI + OpenMP)
 * Run   : mpirun -np 4 ./stencil_mpi N ITERS [--dims R C] [--overlap] [--dump PREFIX EVERY]
 *
 * DECOMPOSITION : MPI_Cart_create on a R x C process grid (MPI_Dims_create picks the squarest by default;
 *                 --dims P 1 gives 1D strips for comparison). Uneven N is handled (first N%R rows get +1).
 * HALOS         : each rank stores (ly+2) x (lx+2) cells; the outer ring holds neighbour data (or the fixed
 *                 boundary = 0 on the global edge, where the neighbour is MPI_PROC_NULL).
 *                 Rows are contiguous; columns use an MPI_Type_vector.
 * MODES         : default  = blocking MPI_Sendrecv halo exchange, then compute all cells.
 *                 --overlap = MPI_Isend/Irecv, compute INTERIOR cells while halos are in flight,
 *                             MPI_Waitall, then compute the one-cell-thick BORDER ring.
 * OUTPUT        : --dump writes the global grid with MPI-IO (MPI_Type_create_subarray) every EVERY sweeps. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "stencil_common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define IDX(i, j) ((long)(i) * W + (j))

int main(int argc, char **argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   /* MPI calls only from main thread */
    int rank, P; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc < 3) { if (!rank) fprintf(stderr, "usage: %s N ITERS [--dims R C] [--overlap] [--dump PREFIX EVERY]\n", argv[0]); MPI_Finalize(); return 1; }
    long N = atol(argv[1]); int iters = atoi(argv[2]);
    int dims[2] = {0, 0}, overlap = 0, dump_every = 0; const char *dump_prefix = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dims") && i + 2 < argc) { dims[0] = atoi(argv[i + 1]); dims[1] = atoi(argv[i + 2]); i += 2; }
        else if (!strcmp(argv[i], "--overlap")) overlap = 1;
        else if (!strcmp(argv[i], "--dump") && i + 2 < argc) { dump_prefix = argv[i + 1]; dump_every = atoi(argv[i + 2]); i += 2; }
    }
    if (dims[0] * dims[1] != 0 && dims[0] * dims[1] != P) { if (!rank) fprintf(stderr, "--dims R C must multiply to np=%d\n", P); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (dims[0] == 0) MPI_Dims_create(P, 2, dims);

    /* ---- Cartesian topology ---- */
    MPI_Comm cart; int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
    MPI_Comm_rank(cart, &rank);
    int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);
    int north, south, west, east;
    MPI_Cart_shift(cart, 0, 1, &north, &south);          /* dim 0 = rows    */
    MPI_Cart_shift(cart, 1, 1, &west, &east);            /* dim 1 = columns */

    /* ---- block sizes (handles N not divisible by dims) ---- */
    long ly = N / dims[0] + (coords[0] < N % dims[0]), lx = N / dims[1] + (coords[1] < N % dims[1]);
    long y0 = coords[0] * (N / dims[0]) + (coords[0] < N % dims[0] ? coords[0] : N % dims[0]);
    long x0 = coords[1] * (N / dims[1]) + (coords[1] < N % dims[1] ? coords[1] : N % dims[1]);
    long W = lx + 2;
    double *a = calloc((ly + 2) * W, sizeof(double)), *b = calloc((ly + 2) * W, sizeof(double));
    if (!a || !b) { fprintf(stderr, "rank %d: out of memory (%.2f GB)\n", rank, 2.0 * (ly + 2) * W * 8 / 1e9); MPI_Abort(MPI_COMM_WORLD, 2); }
    for (long i = 0; i < ly; i++) {
        for (long j = 0; j < lx; j++) {
            a[IDX(i + 1, j + 1)] = init_value(y0 + i, x0 + j, N);
        }
    }

    MPI_Datatype col; MPI_Type_vector((int)ly, 1, (int)W, MPI_DOUBLE, &col); MPI_Type_commit(&col);

    /* ---- MPI-IO view of the global grid for snapshots ---- */
    MPI_Datatype tile_mem, tile_file;
    { int gsz[2] = {(int)N, (int)N}, sub[2] = {(int)ly, (int)lx}, st[2] = {(int)y0, (int)x0};
      MPI_Type_create_subarray(2, gsz, sub, st, MPI_ORDER_C, MPI_DOUBLE, &tile_file); MPI_Type_commit(&tile_file);
      int msz[2] = {(int)(ly + 2), (int)W}, mst[2] = {1, 1};
      MPI_Type_create_subarray(2, msz, sub, mst, MPI_ORDER_C, MPI_DOUBLE, &tile_mem); MPI_Type_commit(&tile_mem); }
    #define DUMP(step) do { if (dump_prefix) { char fn[512]; snprintf(fn, sizeof fn, "%s_%05d.bin", dump_prefix, (step)); \
        MPI_File f; MPI_File_open(cart, fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f); \
        MPI_File_set_view(f, 0, MPI_DOUBLE, tile_file, "native", MPI_INFO_NULL); \
        MPI_File_write_all(f, a, 1, tile_mem, MPI_STATUS_IGNORE); MPI_File_close(&f); } } while (0)

    double comm_time = 0.0;
    double post_time = 0.0;
    double interior_time = 0.0;
    double wait_time = 0.0;
    double border_time = 0.0;
    MPI_Barrier(cart);
    double T0 = MPI_Wtime();
    if (dump_every > 0) DUMP(0);

    for (int it = 0; it < iters; it++) {
        double c0 = MPI_Wtime();
        if (!overlap) {
            /* ---------- blocking exchange: N/S rows then W/E columns ---------- */
            MPI_Sendrecv(&a[IDX(1, 1)],      (int)lx, MPI_DOUBLE, north, 0, &a[IDX(ly + 1, 1)], (int)lx, MPI_DOUBLE, south, 0, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&a[IDX(ly, 1)],     (int)lx, MPI_DOUBLE, south, 1, &a[IDX(0, 1)],      (int)lx, MPI_DOUBLE, north, 1, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&a[IDX(1, 1)],      1, col,  west,  2, &a[IDX(1, lx + 1)], 1, col,  east,  2, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&a[IDX(1, lx)],     1, col,  east,  3, &a[IDX(1, 0)],      1, col,  west,  3, cart, MPI_STATUS_IGNORE);
            comm_time += MPI_Wtime() - c0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long i = 1; i <= ly; i++)
                for (long j = 1; j <= lx; j++)
                    b[IDX(i, j)] = JACOBI(a[IDX(i - 1, j)], a[IDX(i + 1, j)], a[IDX(i, j - 1)], a[IDX(i, j + 1)]);
        } else {
            /* ---------- non-blocking: post, compute interior, wait, compute border ---------- */
            MPI_Request rq[8]; int n = 0;
            double post_start = MPI_Wtime();
            MPI_Irecv(&a[IDX(0, 1)],       (int)lx, MPI_DOUBLE, north, 1, cart, &rq[n++]);
            MPI_Irecv(&a[IDX(ly + 1, 1)],  (int)lx, MPI_DOUBLE, south, 0, cart, &rq[n++]);
            MPI_Irecv(&a[IDX(1, 0)],       1, col, west, 3, cart, &rq[n++]);
            MPI_Irecv(&a[IDX(1, lx + 1)],  1, col, east, 2, cart, &rq[n++]);
            MPI_Isend(&a[IDX(1, 1)],       (int)lx, MPI_DOUBLE, north, 0, cart, &rq[n++]);
            MPI_Isend(&a[IDX(ly, 1)],      (int)lx, MPI_DOUBLE, south, 1, cart, &rq[n++]);
            MPI_Isend(&a[IDX(1, 1)],       1, col, west, 2, cart, &rq[n++]);
            MPI_Isend(&a[IDX(1, lx)],      1, col, east, 3, cart, &rq[n++]);
            comm_time += MPI_Wtime() - c0;
            post_time += MPI_Wtime() - post_start;
           
double interior_start = MPI_Wtime();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for (long i = 2; i <= ly - 1; i++) {
    for (long j = 2; j <= lx - 1; j++) {
        b[IDX(i, j)] = JACOBI(
            a[IDX(i - 1, j)],
            a[IDX(i + 1, j)],
            a[IDX(i, j - 1)],
            a[IDX(i, j + 1)]
        );
    }
}

// Measure the complete interior computation once per iteration
interior_time += MPI_Wtime() - interior_start;

c0 = MPI_Wtime();
MPI_Waitall(n, rq, MPI_STATUSES_IGNORE);


            double wait_elapsed = MPI_Wtime() - c0;

            wait_time += wait_elapsed;
            comm_time += wait_elapsed;

            double border_start = MPI_Wtime();
            for (long j = 1; j <= lx; j++) {                                /* border ring: top & bottom rows */
                b[IDX(1, j)] = JACOBI(a[IDX(0, j)], a[IDX(2, j)], a[IDX(1, j - 1)], a[IDX(1, j + 1)]);
                if (ly > 1) b[IDX(ly, j)] = JACOBI(a[IDX(ly - 1, j)], a[IDX(ly + 1, j)], a[IDX(ly, j - 1)], a[IDX(ly, j + 1)]);
            }
            for (long i = 2; i <= ly - 1; i++) {                            /* left & right columns */
                b[IDX(i, 1)] = JACOBI(a[IDX(i - 1, 1)], a[IDX(i + 1, 1)], a[IDX(i, 0)], a[IDX(i, 2)]);
                if (lx > 1) b[IDX(i, lx)] = JACOBI(a[IDX(i - 1, lx)], a[IDX(i + 1, lx)], a[IDX(i, lx - 1)], a[IDX(i, lx + 1)]);
            }

         // ADD THIS LINE HERE
         border_time += MPI_Wtime() - border_start;
        }
        double *t = a; a = b; b = t;
        if (dump_every > 0 && (it + 1) % dump_every == 0) DUMP(it + 1);
    }
    MPI_Barrier(cart);
    double T1 = MPI_Wtime();
    // Measure elapsed time on this MPI process
double local_elapsed = T1 - T0;

// Obtain the maximum elapsed time across all MPI processes
double max_elapsed = 0.0;

MPI_Reduce(
    &local_elapsed,
    &max_elapsed,
    1,
    MPI_DOUBLE,
    MPI_MAX,
    0,
    cart
);

    /* ---- verification: order-independent checksum + total heat ---- */
    uint64_t cs = 0; double heat = 0;
    for (long i = 0; i < ly; i++) {
        for (long j = 0; j < lx; j++) {
            double v = a[IDX(i + 1, j + 1)];
            cs += cell_hash(y0 + i, x0 + j, N, v);
            heat += v;
        }
    }
    uint64_t cs_all = 0; double heat_all = 0, cmax = 0, csum = 0;
    MPI_Reduce(&cs, &cs_all, 1, MPI_UINT64_T, MPI_SUM, 0, cart);
    MPI_Reduce(&heat, &heat_all, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(&comm_time, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&comm_time, &csum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    
double post_sum = 0.0;
double interior_sum = 0.0;
double wait_sum = 0.0;
double border_sum = 0.0;

MPI_Reduce(&post_time, &post_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
MPI_Reduce(&interior_time, &interior_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
MPI_Reduce(&wait_time, &wait_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
MPI_Reduce(&border_time, &border_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (rank == 0) {
        double tt = max_elapsed;  // Use the maximum elapsed time across all processes
        
printf(
    "STATS impl=%s np=%d dims=%dx%d threads=%d "
    "N=%ld iters=%d mode=%s "
    "total_heat=%.6f checksum=%016llx "
    "time=%.4f comm_max=%.4f comm_avg=%.4f "
    "comm_frac=%.3f "
    "post_avg=%.6f interior_avg=%.6f "
    "wait_avg=%.6f border_avg=%.6f\n",
    threads > 1 ? "hybrid" : "mpi",
    P, dims[0], dims[1], threads,
    N, iters,
    overlap ? "overlap" : "blocking",
    heat_all,
    (unsigned long long)cs_all,
    tt,
    cmax,
    csum / P,
    (csum / P) / tt,
    post_sum / P,
    interior_sum / P,
    wait_sum / P,
    border_sum / P
);

    }
    MPI_Type_free(&col); MPI_Type_free(&tile_mem); MPI_Type_free(&tile_file);
    MPI_Finalize();
    return 0;
}
