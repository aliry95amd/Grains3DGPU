# Grains3DGPU

GPU-accelerated Discrete Element Method (DEM) solver for granular material simulations. Handles non-spherical particles (superquadrics, boxes, cylinders, cones) with Nesterov-accelerated GJK collision detection.

## Building

Requires CUDA Toolkit (11.0+), a C++17 compiler, and Python 3.6+ (optional, for visualization).

```bash
git clone https://github.com/AliRY95/Grains3DGPU
cd Grains3DGPU
make install    # builds xerces, the library, and the main binary
```

To rebuild after code changes:
```bash
make update
```

## Running

Simulations are configured through XML input files. See `Tests/xml/` for examples.

## Tests

```bash
make build-tests && make run-tests
```

See `Tests/README.md` for details.

## Validations

Standalone benchmarks for collision detection, GJK iteration counts, and contact force performance live under `Validations/`. Each has its own `run_all_benchmarks.sh` and `plot.py`.

```bash
make build-validation && make run-validation
```
