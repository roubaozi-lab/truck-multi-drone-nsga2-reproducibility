# Solomon-Based Instance Preprocessing

The experimental instances are derived from the Solomon vehicle routing benchmark instances. The selected instances include c101-c103, r101-r103, and rc101-rc103, representing clustered, random, and mixed customer distributions, respectively.

For each selected Solomon instance, the first N customer nodes are used to construct the corresponding N-customer test case. The depot location, customer coordinates, customer demands, service times, and due dates are inherited from the original Solomon instance and then mapped to the truck-multi-drone cooperative delivery setting.

The preprocessing rules are:

1. Customer coordinates are scaled by a factor of 100 to represent meter-level travel distances.
2. Due dates are scaled by a factor of 10 to maintain consistency with the adopted speed setting.
3. Ready times are set to zero in the computational implementation.
4. Truck travel distance is calculated using the Manhattan metric.
5. Drone flight distance is calculated using the Euclidean metric.
6. The same processed coordinates and time-window settings are used in the heuristic algorithms and the Gurobi benchmark.

The original Solomon benchmark should be cited when using these processed instances.
