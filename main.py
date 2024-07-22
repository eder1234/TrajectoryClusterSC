import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Define the shape context function
def compute_shape_context(trajectory, n_points=50, n_bins_r=5, n_bins_theta=12):
    # Resample trajectory to n_points for time invariance
    t = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, n_points)
    trajectory_resampled = np.column_stack([
        np.interp(t_new, t, trajectory[:, 0]),
        np.interp(t_new, t, trajectory[:, 1])
    ])
    
    # Center the trajectory
    trajectory_centered = trajectory_resampled - np.mean(trajectory_resampled, axis=0)
    
    # Compute pairwise distances (log-r) and angles
    diff = trajectory_centered[:, np.newaxis, :] - trajectory_centered[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])
    
    # Define log-polar bins
    r_edges = np.logspace(np.log10(dists.min() + 1e-6), np.log10(dists.max()), n_bins_r + 1)
    theta_edges = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    
    # Compute shape context for each point
    shape_contexts = []
    for i in range(n_points):
        hist, _, _ = np.histogram2d(
            dists[i, :], angles[i, :],
            bins=[r_edges, theta_edges]
        )
        shape_contexts.append(hist.flatten())
    
    return np.array(shape_contexts)

# Step 2: Define a distance function for shape contexts
def shape_context_distance(sc1, sc2):
    return np.sum(cdist(sc1, sc2, metric='cosine')) / (sc1.shape[0] + sc2.shape[0])

# Step 3: Cluster trajectories
def cluster_trajectories(trajectories, n_clusters=3):
    # Compute shape contexts for all trajectories
    shape_contexts = [compute_shape_context(traj) for traj in trajectories]
    
    # Compute pairwise distances between trajectories
    distances = np.zeros((len(trajectories), len(trajectories)))
    for i in range(len(trajectories)):
        for j in range(i+1, len(trajectories)):
            dist = shape_context_distance(shape_contexts[i], shape_contexts[j])
            distances[i, j] = distances[j, i] = dist
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(distances)
    
    return clusters

# Example usage
# Define some sample trajectories
trajectories = [
    np.column_stack((np.linspace(0, 1, 100), np.sin(np.linspace(0, 2*np.pi, 100)))),
    np.column_stack((np.linspace(0, 1, 80), 0.5 * np.sin(np.linspace(0, 4*np.pi, 80)))),
    np.column_stack((np.linspace(0, 1, 120), np.cos(np.linspace(0, 2*np.pi, 120)))),
    np.column_stack((np.linspace(0, 1, 90), 0.5 * np.cos(np.linspace(0, 4*np.pi, 90))))
]

# Cluster the trajectories
clusters = cluster_trajectories(trajectories, n_clusters=2)

# Visualize the results
plt.figure(figsize=(10, 6))
for i, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], label=f'Trajectory {i+1}, Cluster {clusters[i]}')
plt.legend()
plt.title('Clustered Trajectories')
plt.show()

print("Cluster assignments:", clusters)
