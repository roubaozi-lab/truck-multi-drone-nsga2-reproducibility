# Data Description

This folder contains the computational-result files used to support the tables and figures in the manuscript.

## Folders

- `result_tables/`: CSV files corresponding to the main numerical tables in the manuscript.
- `run_level_metrics/`: run-level performance metrics used for aggregated algorithm comparison and sensitivity analysis.
- `pareto_fronts/`: objective values and service-allocation summaries of the representative Pareto-front analysis.

## Data source

The benchmark instances are derived from the Solomon vehicle routing benchmark instances. The preprocessing rules are described in `docs/solomon_preprocessing.md`.

## Notes

The released files retain the objective values, performance metrics, and summary data needed to verify the reported computational results. The original Solomon benchmark should be cited when using the processed instance settings.
