#!/bin/bash
# Helper script to debug NeighborList tests with cuda-gdb

cd "$(dirname "$0")/build"

# Make sure tests are built
# make -j$(nproc)

# Run cuda-gdb with useful commands
cuda-gdb -ex "set cuda memcheck on" \
         -ex "break updateNeighborList" \
         -ex "run" \
         ./GJKPerformanceTest
