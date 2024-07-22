# TrajectoryClusterSC

TrajectoryClusterSC is a Python library for clustering trajectory data using Shape Context descriptors. It provides a robust method for analyzing and grouping similar trajectories while maintaining invariance to time, rotation, and translation, and preserving sensitivity to scale and shape.

## Features

- Trajectory clustering using Shape Context descriptors
- Invariant to time, rotation, and translation
- Sensitive to scale and shape
- Customizable number of clusters
- Visualizations of clustered trajectories

## Installation

To use TrajectoryClusterSC, first clone this repository:
```
git clone https://github.com/eder1234/TrajectoryClusterSC.git
cd TrajectoryClusterSC
```
Then, install the required dependencies:
```
pip install numpy scipy scikit-learn matplotlib
```
## Usage

Here's a basic example of how to use TrajectoryClusterSC:

```python
import numpy as np
from trajectory_cluster import cluster_trajectories

# Define some sample trajectories
trajectories = [
    np.column_stack((np.linspace(0, 1, 100), np.sin(np.linspace(0, 2*np.pi, 100)))),
    np.column_stack((np.linspace(0, 1, 80), 0.5 * np.sin(np.linspace(0, 4*np.pi, 80)))),
    np.column_stack((np.linspace(0, 1, 120), np.cos(np.linspace(0, 2*np.pi, 120)))),
    np.column_stack((np.linspace(0, 1, 90), 0.5 * np.cos(np.linspace(0, 4*np.pi, 90))))
]

# Cluster the trajectories
clusters = cluster_trajectories(trajectories, n_clusters=2)

print("Cluster assignments:", clusters)
```

## Contributing
Contributions to TrajectoryClusterSC are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Shape Context descriptor method is based on the paper "Shape Context: A New Descriptor for Shape Matching and Object Recognition" by Belongie et al.

## Contact
If you have any questions or feedback, please open an issue on this GitHub repository.
