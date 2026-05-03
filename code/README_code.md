# Code Description

This folder contains the source code used for the computational experiments in the manuscript.

## Files

- `main_algorithm_core.py`: implementation of the improved NSGA-II, traditional NSGA-II, KMeans-initialized NSGA-II, and discrete MOPSO-style baseline.
- `gurobi_three_stage_benchmark.py`: aligned three-stage Gurobi benchmark for small-scale instances.
- `precompute_inputs.py`: preprocessing utilities shared by the Gurobi benchmark.
- `requirements.txt`: Python package requirements.

## Example use

```python
import main_algorithm_core as core
pareto_front, pareto_objs, runtime, history, conv_gen = core.run_improved_nsga2(4201, 100)
```

For the Gurobi benchmark, place the Solomon text files in a folder named `solomon_data` under this directory and run:

```bash
python gurobi_three_stage_benchmark.py
```

Gurobi is optional and requires a valid Gurobi installation and license.
