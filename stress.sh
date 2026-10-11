#!/usr/bin/env bash
# Stress test: grid too big / too slow for your machine.  usage: ./stress.sh N ITERS NP TIMEOUT_SECONDS
# e.g. ./stress.sh 40000 500 8 600   (40000^2 doubles x2 buffers = ~25.6 GB total across ranks)
N=${1:-40000}; IT=${2:-500}; NP=${3:-8}; TMO=${4:-600}
./sysinfo.sh
echo "grid ${N}x${N}, ${IT} sweeps, np=${NP}, memory needed ~ $(python3 -c "print(round(2*($N+2)**2*8/1e9,1))") GB"
S=$(date +%s); timeout "$TMO" mpirun -np "$NP" ./stencil_mpi "$N" "$IT" --overlap; RC=$?; E=$(date +%s)
case $RC in 0) echo "RESULT: finished in $((E-S))s";; 124) echo "RESULT: DNF - killed after waiting ${TMO}s (timeout)";;
  *) echo "RESULT: failed rc=$RC after $((E-S))s (out of memory / MPI abort - see messages above)";; esac
