CFLAGS=-O2 -ffp-contract=off -Wall -Wextra
all: stencil_seq stencil_mpi stencil_hybrid
stencil_seq: stencil_seq.c stencil_common.h
	gcc $(CFLAGS) -o $@ stencil_seq.c -lm
stencil_mpi: stencil_mpi.c stencil_common.h
	mpicc $(CFLAGS) -o $@ stencil_mpi.c -lm
stencil_hybrid: stencil_mpi.c stencil_common.h
	mpicc $(CFLAGS) -fopenmp -o $@ stencil_mpi.c -lm
clean:
	rm -f stencil_seq stencil_mpi stencil_hybrid
