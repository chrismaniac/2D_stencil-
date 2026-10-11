#!/usr/bin/env bash
# Prints the machine specs the assignment asks you to state.
(lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"; free -h | head -2; uname -sr; mpirun --version | head -1) 2>/dev/null \
 || (sysctl -n machdep.cpu.brand_string; echo "cores: $(sysctl -n hw.physicalcpu) physical / $(sysctl -n hw.logicalcpu) logical"; echo "RAM: $(( $(sysctl -n hw.memsize)/1073741824 )) GB"; sw_vers; mpirun --version | head -1)
