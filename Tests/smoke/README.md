# Smoke Tests

Quick verification tests for basic GrainsGPU functionality.

## Running the Tests

```bash
cd Tests/smoke
./run_smoke_tests.sh
```

Or specify a custom binary:
```bash
BINARY=/path/to/grains ./run_smoke_tests.sh
```

## Test Cases

### Manual Tests
- **simple_cpu.xml**: Basic CPU simulation with 10 spherical particles
- **simple_gpu.xml**: Basic GPU simulation with 10 spherical particles

### Parametric Tests
Automatically generated from `test_params.json`. Configure parameter ranges:

```json
{
  "param_ranges": {
    "kn": [10000.0, 50000.0],
    "en": [0.3, 0.5],
    "dt": [0.0001, 0.00005]
  },
  "fixed_params": {
    "sim_type": "Standard",
    "etat": 10.0,
    "num_particles": 10
  }
}
```

Generate tests manually:
```bash
python3 generate_tests.py
```

Generated XML files are stored in `generated/` directory.

## What's Tested

Each test runs for 5 timesteps to verify:
- XML parsing
- Particle initialization
- Collision detection
- Time integration
- Output writing

## Expected Runtime

~5-10 seconds per test (depends on parameter combinations).

