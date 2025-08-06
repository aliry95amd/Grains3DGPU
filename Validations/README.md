# Physical Analysis Tools

This directory contains standalone analysis scripts for various physical phenomena studies in the GrainsGPU project.

## Directory Structure

### `convergenceAnalysis/`
Scripts for analyzing numerical convergence properties:
- Forward Euler time integration convergence studies
- Timestep sensitivity analysis
- Error accumulation tracking
- Stability analysis for different integration schemes

### `contactForces/`
Scripts for contact force modeling and analysis:
- Sphere overlap ODE solvers
- Contact force law validation
- Spring-damper model analysis
- Force-displacement relationship studies

### `visualization/`
Scripts for generating plots and visualizations:
- Convergence plots and error analysis charts
- Force evolution graphs
- Energy conservation plots
- Animation utilities for dynamic systems

### `data/`
Directory for storing analysis results:
- Raw simulation output data
- Processed analysis results
- Generated plots and figures
- Configuration files for analysis parameters

## Usage Guidelines

1. **Standalone Scripts**: Each script should be self-contained and executable independently
2. **Data Organization**: Store input data in `data/input/` and results in `data/output/`
3. **Modular Design**: Use common utility functions across scripts
4. **Documentation**: Include clear documentation and usage examples in each script
5. **Parameterization**: Use configuration files for easy parameter adjustment

## Example Workflow

1. Run convergence analysis: `python convergenceAnalysis/forward_euler_convergence.py`
2. Analyze contact forces: `python contactForces/sphere_overlap_ode.py`
3. Generate visualizations: `python visualization/plot_convergence_results.py`

## Dependencies

- Python 3.x with NumPy, SciPy, Matplotlib
- Optional: Pandas for data manipulation
- Optional: Jupyter notebooks for interactive analysis
