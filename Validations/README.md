# Validations

Performance benchmarks and correctness checks for individual modules.

- **CollisionDetection/** -- Broad-phase collision detection throughput (linked-cell variants).
- **CollisionDetectionModule/** -- Full collision pipeline benchmarks.
- **ContactForces/** -- Contact table lookup and force computation timing.
- **GJKIterations/** -- GJK convergence iteration counts across shape pairs and configurations.
- **StaticPacking/** -- Static packing simulation for end-to-end sanity.

Each benchmark directory contains a `main.cpp`, a `Makefile`, a `run_all_benchmarks.sh` script, and a `plot.py` for generating result figures.

## Running

From the repo root:
```bash
make build-validation && make run-validation
```

Or individually:
```bash
cd Validations/CollisionDetection
make
./run_all_benchmarks.sh
python3 plot.py
```
