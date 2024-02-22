#Apply perfect version of DBSCAN
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

CLUSTERING=True


df=pd.read_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/stops_all_uk.csv')


# Assuming 'df' is your DataFrame containing latitude and longitude columns
# Convert latitude and longitude columns to radians before passing to DBSCAN
df_radians = np.radians(df[['latitude', 'longitude']])

# Create DBSCAN object with haversine metric
db = DBSCAN(eps=0.2/6371., min_samples=10, algorithm='ball_tree', metric='haversine').fit(df_radians)

# Extract cluster labels
cluster_labels = db.labels_

# Add cluster labels to the original DataFrame
df['cluster_label'] = cluster_labels

# Print the number of clusters (-1 indicates noise points)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters: {num_clusters}")

# Print the number of noise points
num_noise_points = list(cluster_labels).count(-1)
print(f"Number of noise points: {num_noise_points}")

# Print the DataFrame with cluster labels
print(df)
df=df[df['cluster_label']!=-1]
df.to_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/prova.csv')



# Select only clusters with labels not equal to -1 (noise points)
df_filtered = df[df['cluster_label'] != -1]

# Count the occurrences of each cluster label
cluster_counts = df_filtered['cluster_label'].value_counts()

# Select the 30 most populous clusters
top_30_clusters = cluster_counts.head(50).index

# Filter the DataFrame to include only the top 30 clusters
df_top_clusters = df_filtered[df_filtered['cluster_label'].isin(top_30_clusters)]

# Save the filtered DataFrame to a CSV file
df_top_clusters.to_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/top_50_density_zones.csv')


ports=pd.read_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/PORTS.csv')





centroids=get_cluster_centroids_short(df_top_clusters,details=False)
centroids.to_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/top50.csv')
#new_ports=from_pd_to_gpd(centroids_far_from_ports_short(centroids, ports))# remove redundant ports
#new_ports=remove_offshore_ports(new_ports)