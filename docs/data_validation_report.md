# Data Validation Report

This file summarizes the consistency checks between the released CSV files and the numerical values reported in the manuscript.

## Table 5

`data/result_tables/table5_parameter_configuration_summary.csv` contains the nine Taguchi L9 parameter groups. The values match the manuscript table after sorting by `ExpID`. Exp3 has the highest selection score, 0.7950, with parameters `(Pc, Pm, Npop, ER) = (0.6, 0.15, 100, 0.15)`.

## Pareto analysis and Table 6

`data/pareto_fronts/r101_N100_global_pareto_objectives.csv` contains 94 non-dominated objective-value rows for the representative r101, N=100 Pareto analysis. The minimum total cost is 132.66 USD, the minimum maximum completion time is 1260 s, and the minimum service delay is 0 s. `data/result_tables/table6_representative_solutions.csv` contains the five representative non-dominated solutions reported in the manuscript.

## Table 7

`data/result_tables/table7_small_scale_comparison.csv` contains the small-scale Gurobi and improved NSGA-II comparison values reported in the manuscript, including Gurobi status, Gurobi cost, Tmax lower/upper bounds, gap, NSGA-II cost, NSGA-II Tmax/delay, and runtime.

## Table 8

`data/result_tables/table8_algorithm_comparison_summary.csv` is aggregated from `data/run_level_metrics/table8_run_level_metrics_report_all.csv`. The summary matches the manuscript values. For N=50, the improved NSGA-II has HV_Mean = 4.492e12, runtime = 11.04 s, cost = 77.00 USD, time = 1322.84 s, Pen_Min_Mean = 0, and ZeroPenalty_Ratio_Mean = 0.7316. For N=100, the improved NSGA-II has HV_Mean = 4.135e12, runtime = 19.46 s, cost = 147.14 USD, time = 1498.77 s, Pen_Min_Mean = 0, and ZeroPenalty_Ratio_Mean = 0.6609.

## Table 9

`data/result_tables/table9_mode_comparison_by_instance.csv` and `data/result_tables/table9_mode_comparison_overall.csv` match the manuscript's cooperative-mode comparison. The overall average cost of Truck-only is 179.20 USD with average service delay of 420.33 s and average completion time of 1723.00 s. Among zero-delay modes, Full cooperative has the lowest average cost, 182.73 USD, and a competitive average completion time of 1647.33 s.

## Tables 10 and 11

`data/result_tables/table10_truck_count_sensitivity.csv` and `data/result_tables/table11_speed_sensitivity.csv` are aggregated from `data/run_level_metrics/sensitivity_run_level_metrics.csv`. For truck-count sensitivity, average cost increases from 103.33 USD at 6 trucks to 215.31 USD at 15 trucks, while average completion time decreases from 2225.67 s to 1495.33 s. Average service delay becomes zero when the truck count reaches 10. For speed sensitivity, the baseline configuration is 10/20 m/s, and the compared speed settings are 10/25, 15/20, and 15/25 m/s.
