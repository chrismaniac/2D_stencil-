/* stencil_common.h - shared by sequential and MPI versions: initial condition + order-independent checksum */
#ifndef STENCIL_COMMON_H
#define STENCIL_COMMON_H
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Initial heat: two hot discs on a cold plate. Global boundary (halo ring) is fixed at 0. */
static inline double init_value(long gi, long gj, long N) {
    double cy = N / 2.0, cx = N / 2.0, r = N / 8.0;
    double dy = gi - cy, dx = gj - cx;
    if (dy * dy + dx * dx <= r * r) return 100.0;
    double cy2 = N / 4.0, cx2 = N / 4.0, r2 = N / 16.0;
    dy = gi - cy2; dx = gj - cx2;
    if (dy * dy + dx * dx <= r2 * r2) return 50.0;
    return 0.0;
}

/* One Jacobi update. IDENTICAL expression in seq and MPI => bit-identical results for any decomposition. */
#define JACOBI(up, down, left, right) (0.25 * (((up) + (down)) + ((left) + (right))))

/* Order-independent checksum: wrapping sum of a hash of (cell index, bit pattern). */
static inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31);
}
static inline uint64_t cell_hash(long gi, long gj, long N, double v) {
    uint64_t b; memcpy(&b, &v, sizeof b);
    return mix64(b ^ mix64((uint64_t)(gi * N + gj)));
}
#endif
