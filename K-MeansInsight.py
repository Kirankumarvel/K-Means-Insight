import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic dataset with 4 clusters
X, y = make_blobs(n_samples=5000, 
                   centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], 
                   cluster_std=0.9, 
                   random_state=42)

# Check dataset shape
print("Feature Matrix Shape:", X.shape)
print("Response Vector Shape:", y.shape)

plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)
plt.title("Scatter Plot of Randomly Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)  # Add grid for better readability
plt.show()


KMEANS_INIT = "k-means++"  # Define a constant for k-means initialization
k_means = KMeans(init=KMEANS_INIT, n_clusters=4, n_init=12, random_state=0)

# Fit the K-Means model to the dataset
k_means.fit(X)


# Get cluster labels for each data point
k_means_labels = k_means.labels_

# Display unique labels (clusters) and preview first 10 labels
print("Unique cluster labels:", np.unique(k_means_labels))
print("First 10 cluster labels:", k_means_labels[:10])

k_means_cluster_centers = k_means.cluster_centers_

# Display the cluster centers
print("Cluster Centers:\n", k_means_cluster_centers)

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Generate a color map for different clusters
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# Loop to plot data points and centroids
for k, col in zip(range(len(k_means_cluster_centers)), colors):
    # Boolean mask for cluster membership
    my_members = (k_means_labels == k)

    # Define the centroid for this cluster
    cluster_center = k_means_cluster_centers[k]

    # Plot the data points
    ax.scatter(X[my_members, 0], X[my_members, 1], color=col, marker='o', s=10, alpha=0.6)

    # Plot the centroids
    ax.scatter(cluster_center[0], cluster_center[1], marker='o', color=col, edgecolor='k', s=200, linewidth=2)

# Title of the plot
ax.set_title('KMeans Clustering')

# Remove x-axis and y-axis ticks
ax.set_xticks(())
ax.set_yticks(())

# Show the plot
plt.show()


#Task 1: Try to cluster the above dataset into a different number of clusters, say k=3. Note the difference in the pattern generated.
k_means_3 = KMeans(init=KMEANS_INIT, n_clusters=3, n_init=12, random_state=42)
# Initialize KMeans with k=3
k_means_3 = KMeans(init=KMEANS_INIT, n_clusters=3, n_init=12)
k_means_3 = KMeans(init=KMEANS_INIT, n_clusters=3, n_init=12, random_state=42)
# Fit the model with the feature matrix X
k_means_3.fit(X)

# Get the labels for each point
k_means_labels_3 = k_means_3.labels_

# Get the new cluster centers
k_means_cluster_centers_3 = k_means_3.cluster_centers_

# Initialize the plot
fig = plt.figure(figsize=(6, 4))

# Generate color map
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_3))))

# Create a subplot
ax = fig.add_subplot(1, 1, 1)

# Loop to plot data points and centroids
for k, col in zip(range(len(k_means_cluster_centers_3)), colors):
    # Boolean mask for cluster membership
    my_members = (k_means_labels_3 == k)

    # Define the new cluster center
    cluster_center = k_means_cluster_centers_3[k]

    # Plot data points
    ax.scatter(X[my_members, 0], X[my_members, 1], color=col, marker='o', s=10, alpha=0.6)

    # Plot centroids
    ax.scatter(cluster_center[0], cluster_center[1], marker='o', color=col, edgecolor='k', s=200, linewidth=2)

# Set the title
ax.set_title('KMeans Clustering (k=3)')

# Remove x and y ticks
ax.set_xticks(())
ax.set_yticks(())

# Show plot
plt.show()


k_means_5 = KMeans(init=KMEANS_INIT, n_clusters=5, n_init=12, random_state=42)
# Initialize KMeans with k=5
k_means_5 = KMeans(init=KMEANS_INIT, n_clusters=5, n_init=12)
k_means_5 = KMeans(init=KMEANS_INIT, n_clusters=5, n_init=12, random_state=42)
# Fit the model with the feature matrix X
k_means_5.fit(X)

# Get the labels for each point
k_means_labels_5 = k_means_5.labels_

# Get the new cluster centers
k_means_cluster_centers_5 = k_means_5.cluster_centers_

# Initialize the plot
fig = plt.figure(figsize=(6, 4))

# Generate color map
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_5))))

# Create a subplot
ax = fig.add_subplot(1, 1, 1)

# Loop to plot data points and centroids
for k, col in zip(range(len(k_means_cluster_centers_5)), colors):
    # Boolean mask for cluster membership
    my_members = (k_means_labels_5 == k)

    # Define the new cluster center
    cluster_center = k_means_cluster_centers_5[k]

    # Plot data points
    ax.scatter(X[my_members, 0], X[my_members, 1], color=col, marker='o', s=10, alpha=0.6)

    # Plot centroids
    ax.scatter(cluster_center[0], cluster_center[1], marker='o', color=col, edgecolor='k', s=200, linewidth=2)

# Set the title
ax.set_title('KMeans Clustering (k=5)')

# Remove x and y ticks
ax.set_xticks(())
ax.set_yticks(())

# Show plot
plt.show()


# Removed unused statement
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df
# Removed unused statement
#Pre-processing
cust_df = cust_df.drop('Address', axis=1)
# Drop NaNs from the dataframe
cust_df = cust_df.dropna()
cust_df.info()

X = cust_df.drop('Customer Id', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head()

#Normalizing over the standard deviation

#task 3: Write the code to cluster the data with k=3. Extract the cluster labels for this clustering process.
# Define the number of clusters
clusterNum = 3

# Initialize KMeans with k=3
k_means = KMeans(n_clusters=clusterNum, init=KMEANS_INIT, n_init=12, random_state=42)

# Fit the model to the standardized dataset
k_means.fit(X_scaled)

# Extract cluster labels
labels = k_means.labels_

# Print first 10 labels for verification
print(labels[:10])

cust_df["Clus_km"] = labels

cust_df.groupby('Clus_km').mean()
centroids = cust_df.groupby("Clus_km").mean()
print(centroids)
area = np.pi * (X_scaled.iloc[:, 1] / X_scaled.iloc[:, 1].max())**2  # Normalize for better visualization
# Create scatter plot
X = cust_df.iloc[:, 1:].values
area = np.pi * (X[:, 1] / X[:, 1].max())**2  # Normalize for better visualization

plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 3], s=area * 50, c=labels.astype(float), cmap='tab10', ec='k', alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.title('Customer Segmentation based on Age and Income', fontsize=18)
plt.show()


# Create interactive 3D scatter plot
fig = px.scatter_3d(
    x=X[:, 1],  # Education
    y=X[:, 0],  # Age
    z=X[:, 3],  # Income
    opacity=0.7,
    color=labels.astype(str),  # Convert labels to string for better categorical coloring
)

fig.update_traces(marker=dict(size=5, line=dict(width=0.5)), showlegend=True)
fig.update_layout(
    coloraxis_showscale=False, 
    width=1000, height=800, 
    scene=dict(
        xaxis=dict(title="Education"),
        yaxis=dict(title="Age"),
        zaxis=dict(title="Income")
    )
)

fig.show()

