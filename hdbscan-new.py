import ast
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates to polar coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

# Step 1: Read the text file and extract all tuples
file_path = 'outputs/6.txt'  # Change this to your text file path
all_points = []

with open(file_path, 'r') as file:
    for line in file:
        tuples = ast.literal_eval(line.strip())
        for t in tuples:
            _, x, y = t
            all_points.append((x, y))

# Step 2: Convert the points to polar coordinates (r, theta)
x_coords, y_coords = zip(*all_points)
polar_coords = [cartesian_to_polar(x, y) for x, y in zip(x_coords, y_coords)]

# Step 3: Prepare data for HDBSCAN clustering
data = np.array(polar_coords)  # Convert to numpy array
r_values = data[:, 0]
theta_values = data[:, 1]

# Step 4: Apply HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='cityblock', cluster_selection_method='eom')#, metric='cityblock', cluster_selection_method='eom'
labels = clusterer.fit_predict(data)

# Step 5: Plot the clustered points in polar coordinates
plt.figure(figsize=(10, 8))
scatter = plt.scatter(theta_values, r_values, c=labels, cmap='viridis', s=50)
plt.colorbar(scatter, label='Cluster Label')
plt.title('HDBSCAN Clustering in Polar Coordinates')
plt.xlabel('Angle (radians)')
plt.ylabel('Distance')
plt.grid(True)
# plt.show()
plt.savefig('hdbscan_clusters_polar.png')  # Save the plot as an image file

# Step 6: Optionally, plot the clustered points in Cartesian coordinates
plt.figure(figsize=(10, 8))
scatter_cartesian = plt.scatter(x_coords, y_coords, c=labels, cmap='viridis', s=50)
plt.colorbar(scatter_cartesian, label='Cluster Label')
plt.title('HDBSCAN Clustering in Cartesian Coordinates')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
# plt.show()
plt.savefig('hdbscan_clusters.png')  # Save the plot as an image file

#Print the number of clusters
print("Number of clusters:", len(set(labels)))

# Step 7: Write the number of clusters to Firestore
import write_to_db
write_to_db.write(len(set(labels)))  # Call the write function from write_to_firestore.py
