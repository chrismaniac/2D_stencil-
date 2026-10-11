/* stencil_seq.c - sequential 2D heat diffusion (Jacobi, 5-point). 
 * Build: gcc -O2 -ffp-contract=off -o stencil_seq stencil_seq.c -lm
 * Run  : ./stencil_seq N ITERS            (N x N interior cells, ITERS Jacobi sweeps) */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stencil_common.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s N ITERS\n", argv[0]); return 1; }
    long N = atol(argv[1]); int iters = atoi(argv[2]); long W = N + 2;
    double *a = calloc(W * W, sizeof(double)), *b = calloc(W * W, sizeof(double));
    if (!a || !b) { fprintf(stderr, "out of memory (%.1f GB needed)\n", 2.0 * W * W * 8 / 1e9); return 2; }
    for (long i = 0; i < N; i++) for (long j = 0; j < N; j++) a[(i + 1) * W + j + 1] = init_value(i, j, N);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (long i = 1; i <= N; i++)
            for (long j = 1; j <= N; j++)
                b[i * W + j] = JACOBI(a[(i - 1) * W + j], a[(i + 1) * W + j], a[i * W + j - 1], a[i * W + j + 1]);
        double *t = a; a = b; b = t;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    uint64_t cs = 0; double total = 0;
    for (long i = 0; i < N; i++) for (long j = 0; j < N; j++) { double v = a[(i + 1) * W + j + 1]; cs += cell_hash(i, j, N, v); total += v; }
    printf("STATS impl=seq np=1 N=%ld iters=%d total_heat=%.6f checksum=%016llx time=%.4f\n", N, iters, total, (unsigned long long)cs, el);
    return 0;
}
