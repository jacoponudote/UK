"""## Imports and Load"""
import shapely
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os 

# From a folder create the file paths dataframe
def create_file_paths_dataframe(folder_path):
    """
    Create a DataFrame listing all file paths from files within a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing files.

    Returns:
    - pd.DataFrame: DataFrame with a column named "File Paths" listing all file paths.
    """
    # Ensure the folder path is valid
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Invalid folder path - {folder_path}")
        return pd.DataFrame()

    # Get a list of all files in the folder
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # Create a DataFrame with a column "File Paths"
    file_paths_df = pd.DataFrame({'File_Path': files})

    return file_paths_df
def filter_uk_coordinates(df):
    # Converti le colonne 'longitude' e 'latitude' in numeri
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    uk_longitude_min = -10
    uk_longitude_max = 2
    uk_latitude_min = 49
    uk_latitude_max = 61

    uk_data = df[
        (df['longitude'] >= uk_longitude_min) & (df['longitude'] <= uk_longitude_max) &
        (df['latitude'] >= uk_latitude_min) & (df['latitude'] <= uk_latitude_max)
    ]

    return uk_data
# Calculate the distance between two points given their latitude and longitude
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance* 0.539957

# Calculate from ais file of a ship, all the stops and return a pandas df
def stops_detection(ais):
    filtered_df = ais[(ais['observed_sog'] < 3) & (ais['navigational_status'] == 'Moored')]
    filtered_df = filtered_df[['imo', 'longitude', 'latitude']]  # Keep only required columns
    return filtered_df

# From a folder full of AIS data to a dataframe with stops
def process_files(files_path, start, stop, stops_detection_function, id_output):
    # Read the CSV file
    files = pd.read_csv(files_path)
    files = files['File_Path']

    # Initialize an empty list to store rows
    all_rows = []

    # Process files
    for file_path in files[start:stop]:
        try:
            ais = pd.read_csv(file_path, usecols=['longitude', 'latitude', 'imo', 'navigational_status','sog'])
            # Apply the stops_detection function with ais as an argument
            stops_result = stops_detection_function(ais)  # Replace with your actual function call

            # Append the output rows to all_rows list
            all_rows.append(stops_result)
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping...")

    # Concatenate the outputs
    concatenated_output = pd.concat(all_rows)
    concatenated_output.to_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/stops/' + id_output + '.csv', index=False)
    return concatenated_output

def process_cluster(cluster, result_data, cluster_sizes, min_cluster_size, max_cluster_size):
    cluster_size = len(cluster)-1
    if min_cluster_size <= cluster_size <= max_cluster_size:
        result_data.append({
            'longitude': cluster[-1]['longitude'],
            'latitude': cluster[-1]['latitude'],
            'imo': int(cluster[-1]['imo']),
        })
    return []

def stops_detection(ais, min_cluster_size=2, max_cluster_size=1000, num_workers=None):
    clusters = []
    current_cluster = []

    result_data = []
    cluster_sizes = []

    for index, row in ais.iterrows():
        if len(current_cluster) == 0:
            current_cluster.append(row)
        else:
            last_row = current_cluster[-1]
            if  (float(row['sog']) < 3):
                current_cluster.append(row)
            else:
                clusters.extend(process_cluster(current_cluster, result_data, cluster_sizes, min_cluster_size, max_cluster_size))
                current_cluster = [row]

    clusters.append(current_cluster)  # Append the last cluster

    stops_detection_df = pd.DataFrame()
    if num_workers is None or num_workers == 1:
        for cluster in clusters:
            process_cluster(cluster, result_data, cluster_sizes, min_cluster_size, max_cluster_size)
    else:
        process_cluster(clusters, result_data, cluster_sizes, min_cluster_size, max_cluster_size)

    stops_detection_df = pd.DataFrame(result_data)
    return stops_detection_df

def write_to_csv(writer, row, file_name):
    try:
        writer.writerow(row)
    except csv.Error as e:
        warnings.warn(f"Error writing row to '{file_name}': {str(e)}", UserWarning)

def process_csv_file(file_path, writer):
    with open(file_path, 'rb') as in_file:
        for raw_line in in_file:
            # Decode the line, ignoring errors
            decoded_line = raw_line.decode('utf-8', errors='ignore').rstrip('\r\n')

            # Split the line into fields and write to CSV
            row = decoded_line.split(',')
            write_to_csv(writer, row, file_path)
            
            
# Import necessary libraries
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import shapely.geometry
from tqdm import tqdm
import pycountry
import reverse_geocoder as rg
from shapely.geometry import Polygon, Point
import pycountry
import reverse_geocoder as rg
import iso3166
from pprint import pprint
import os
from shapely import wkt
root='G:/My Drive/Port Detection Algorithm/'

# From a folder create the file paths dataframe
def create_file_paths_dataframe(folder_path):
    """
    Create a DataFrame listing all file paths from files within a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing files.

    Returns:
    - pd.DataFrame: DataFrame with a column named "File Paths" listing all file paths.
    """
    # Ensure the folder path is valid
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Invalid folder path - {folder_path}")
        return pd.DataFrame()

    # Get a list of all files in the folder
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # Create a DataFrame with a column "File Paths"
    file_paths_df = pd.DataFrame({'File Paths': files})

    return file_paths_df

# Calculate the distance (in nm) between two points given their latitude and longitude
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two sets of latitude and longitude coordinates.

    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.

    Returns:
        The haversine distance between the two points in nautical miles.
    """
    R = 6371  # Earth radius in kilometers
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance * 0.539957  # Convert to nautical miles

# Compute clustering of points using DBSCAN and plot points with colored by cluster
def dbscan(stops, eps=0.001, min_samples=100, metric='haversine'):
    """
    Perform DBSCAN clustering on a set of stop points.

    Args:
        stops: A DataFrame containing stop points with 'LATITUDE' and 'LONGITUDE' columns.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        metric: The distance metric to use for clustering ('haversine' for geographic coordinates).

    Returns:
        A DataFrame with clustered points and a 'CLUSTER_LABEL' column indicating the cluster label for each point.
    """
    stops = stops.dropna()
    X = stops[['LATITUDE', 'LONGITUDE']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    stops['CLUSTER_LABEL'] = clusters.labels_
    return stops

# Convert a pandas dataframe to a geopandas dataframe
def from_pd_to_gpd(input_df):
    """
    Convert a pandas DataFrame to a GeoPandas GeoDataFrame.

    Args:
        input_df: The input pandas DataFrame with 'LONGITUDE' and 'LATITUDE' columns.

    Returns:
        A GeoDataFrame with geometry column.
    """
    geometry = [shapely.geometry.Point(xy) for xy in zip(input_df['longitude'].astype(float), input_df['latitude'].astype(float))]
    input_df['geometry'] = geometry
    return input_df

# Get cluster centroids with maximum distance within each cluster
def get_cluster_centroids(df, details=False):
    """
    Calculates the cluster centroids and maximum distance between points in each cluster.

    Args:
        df: The DataFrame containing the data.
        details: Whether to calculate the maximum distance between points in each cluster.

    Returns:
        A DataFrame containing the cluster centroids and maximum distance between points in each cluster.
    """
    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
    df['LATITUDE'] = df['LATITUDE'].astype(float)
    aggregated = df.groupby('CLUSTER_LABEL').agg({
        'LONGITUDE': 'mean',
        'LATITUDE': 'mean',
    }).reset_index()

    if details:
        max_distances = []

        for cluster_label, cluster_data in tqdm(df.groupby('CLUSTER_LABEL')):
            if cluster_data.shape[0] > 100:
                cluster_points = cluster_data[['LATITUDE', 'LONGITUDE']].sample(100, random_state=0).values
            else:
                cluster_points = cluster_data[['LATITUDE', 'LONGITUDE']].values

            distances = haversine_distance(
                cluster_points[:, 0][:, np.newaxis],
                cluster_points[:, 1][:, np.newaxis],
                cluster_points[:, 0],
                cluster_points[:, 1]
            )

            max_distance = distances.max()
            max_distances.append(max_distance)

        aggregated['MAX_DISTANCE'] = max_distances

    return aggregated
# Get cluster centroids with maximum distance within each cluster
def get_cluster_centroids_short(df, details=False):
    """
    Calculates the cluster centroids and maximum distance between points in each cluster.

    Args:
        df: The DataFrame containing the data.
        details: Whether to calculate the maximum distance between points in each cluster.

    Returns:
        A DataFrame containing the cluster centroids and maximum distance between points in each cluster.
    """
    df['longitude'] = df['longitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    aggregated = df.groupby('cluster_label').agg({
        'longitude': 'mean',
        'latitude': 'mean',
    }).reset_index()

    if details:
        max_distances = []

        for cluster_label, cluster_data in tqdm(df.groupby('cluster_label')):
            if cluster_data.shape[0] > 100:
                cluster_points = cluster_data[['latitude', 'longitude']].sample(100, random_state=0).values
            else:
                cluster_points = cluster_data[['latitude', 'longitude']].values

            distances = haversine_distance(
                cluster_points[:, 0][:, np.newaxis],
                cluster_points[:, 1][:, np.newaxis],
                cluster_points[:, 0],
                cluster_points[:, 1]
            )

            max_distance = distances.max()
            max_distances.append(max_distance)

        aggregated['MAX_DISTANCE'] = max_distances

    return aggregated

# Find centroids far from ports
def centroids_far_from_ports(centroids, ports):
    """
    Filter centroids that are far from ports based on distance threshold.

    Args:
        centroids: A DataFrame containing cluster centroids with 'LATITUDE' and 'LONGITUDE' columns.
        ports: A DataFrame containing port information with 'LATITUDE' and 'LONGITUDE' columns.

    Returns:
        A filtered DataFrame containing centroids far from ports.
    """
    filtered_rows = []
    for _, centroid in tqdm(centroids.iterrows(), total=len(centroids), desc="Processing centroids"):
        id = centroid['cluster_label']
        lat_c, lon_c = centroid['latitude'], centroid['longitude']
        ports_filtered = ports[
            (abs(lat_c - ports['LATITUDE'].astype(float)) <= 0.5) |
            (abs(lon_c - ports['LONGITUDE'].astype(float)) <= 0.5)
        ]
        far_from_all_ports = ports_filtered.apply(
            lambda port: haversine_distance(lat_c, lon_c, port['LATITUDE'], port['LONGITUDE']) >.0000001  ,
            axis=1
        ).all()
        if far_from_all_ports:
            filtered_rows.append(id)
    filtered_df = centroids[centroids['cluster_label'].isin(filtered_rows)]
    return filtered_df 

def centroids_far_from_ports_short(centroids, ports):
    """
    Filter centroids that are far from ports based on distance threshold.

    Args:
        centroids: A DataFrame containing cluster centroids with 'LATITUDE' and 'LONGITUDE' columns.
        ports: A DataFrame containing port information with 'LATITUDE' and 'LONGITUDE' columns.

    Returns:
        A filtered DataFrame containing centroids far from ports.
    """
    filtered_rows = []
    for _, centroid in tqdm(centroids.iterrows(), total=len(centroids), desc="Processing centroids"):
        id = centroid['cluster_label']
        lat_c, lon_c = centroid['latitude'], centroid['longitude']
        ports_filtered = ports[
            (abs(lat_c - ports['LATITUDE'].astype(float)) <= 0.5) |
            (abs(lon_c - ports['LONGITUDE'].astype(float)) <= 0.5)
        ]
        far_from_all_ports = ports_filtered.apply(
            lambda port: haversine_distance(lat_c, lon_c, port['LATITUDE'], port['LONGITUDE']) >5  ,
            axis=1
        ).all()
        if far_from_all_ports:
            filtered_rows.append(id)
    filtered_df = centroids[centroids['cluster_label'].isin(filtered_rows)]
    return filtered_df 

# Percentage of NEW_PORTS detected by Valentin (with ROUTE_info.py) covered using the algorithm detection
def accuracy_NEW_PORTS(new_ports, selected_columns, fast=True):
    """
    Calculate the accuracy of NEW_PORTS detection.

    Args:
        new_ports (DataFrame): DataFrame containing information about new ports.
        selected_columns (DataFrame): DataFrame containing selected columns.
        fast (bool): If True, use a subset of selected columns for faster calculation.

    Returns:
        float: Accuracy percentage.
    """
    combined_ports = new_ports  # Assuming combined_ports is the same as new_ports

    if fast:
        selected_columns = selected_columns.sample(1000)  # Sample a subset for faster processing

    results = []

    # Iterate over selected_columns with a progress bar
    for _, V in tqdm(selected_columns.iterrows(), total=len(selected_columns), desc="Calculating Accuracy"):
        lon = V['LONGITUDE']
        lat = V['LATITUDE']

        # Filter ports within a certain range of coordinates
        ports_filtered = combined_ports[
            (abs(lat - combined_ports['LATITUDE'].astype(float)) <= 0.2) |
            (abs(lon - combined_ports['LONGITUDE'].astype(float)) <= 0.2)
        ]

        # Check if any port is close to the current coordinate
        far_from_all_ports = ports_filtered.apply(
            lambda port: haversine_distance(lat, lon, port['LATITUDE'], port['LONGITUDE']) < 4,
            axis=1
        ).any()

        results.append(far_from_all_ports)

    accuracy = sum(results) / len(results) * 100
    return accuracy








def add_port_coordinates_to_points(points_gdf, ports_gdf):
    """
    Add port coordinates to points GeoDataFrame.

    Args:
        points_gdf (GeoDataFrame): Points GeoDataFrame.
        ports_gdf (GeoDataFrame): Ports GeoDataFrame.

    Returns:
        GeoDataFrame: Points GeoDataFrame with port coordinates.
    """
    # Create empty columns for port coordinates
    points_gdf['closest_port_latitude'] = np.nan
    points_gdf['closest_port_longitude'] = np.nan
    points_gdf['CNTR_CODE3_port'] = np.nan
    points_gdf['PORT_NAME_port'] = np.nan

    # Build a spatial index for ports_gdf
    ports_sindex = ports_gdf.sindex

    for i, point in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0]):
        lon = float(point['LONGITUDE'])
        lat = float(point['LATITUDE'])

        # Create a bounding box around the point for spatial query
        bbox = (lon - 0.2175, lat - 0.2175, lon + 0.2175, lat + 0.2175)

        # Query the ports_sindex for ports within the bounding box
        possible_ports_index = list(ports_sindex.intersection(bbox))

        if possible_ports_index:
            # Filter ports by index
            possible_ports = ports_gdf.iloc[possible_ports_index]

            # Calculate distances to the possible ports and get the closest one
            distances = possible_ports.apply(
                lambda row: ((lat - float(row['LATITUDE'])) ** 2 + (lon - float(row['LONGITUDE'])) ** 2) ** 0.5,
                axis=1
            )
            closest_port_index = distances.idxmin()

            closest_port = possible_ports.loc[closest_port_index]

            # Extract the coordinates from the Point geometry
            closest_port_lat = float(closest_port['geometry'].y)
            closest_port_lon = float(closest_port['geometry'].x)

            # Add the port coordinates to the points GeoDataFrame
            points_gdf.at[i, 'closest_port_latitude'] = closest_port_lat
            points_gdf.at[i, 'closest_port_longitude'] = closest_port_lon
            points_gdf.at[i, 'CNTR_CODE3_port'] = closest_port['CNTR_CODE3']
            points_gdf.at[i, 'PORT_NAME_port'] = closest_port['PORT_NAME']

    return points_gdf




import geopandas as gpd
from geopy.distance import great_circle


import geopandas as gpd
from geopy.distance import great_circle
from tqdm import tqdm
import random
from typing import Optional

def far_from_ports(centroids_gdf: gpd.GeoDataFrame, ports_gdf: gpd.GeoDataFrame,
                   lon_threshold: float = 0.05, lat_threshold: float = 0.05,
                   distance_threshold: float = 0.0001) -> gpd.GeoDataFrame:
    """
    Filters centroids that are far from ports based on distance and location thresholds.

    Args:
        centroids_gdf: GeoDataFrame containing centroid data.
        ports_gdf: GeoDataFrame containing port data.
        lon_threshold: Longitude threshold for filtering close centroids.
        lat_threshold: Latitude threshold for filtering close centroids.
        distance_threshold: Distance threshold for considering a centroid as far from ports.

    Returns:
        A GeoDataFrame containing the filtered centroids.
    """
    # Convert latitude and longitude columns to numeric
    centroids_gdf['LATITUDE'] = centroids_gdf['LATITUDE'].astype(float)
    centroids_gdf['LONGITUDE'] = centroids_gdf['LONGITUDE'].astype(float)
    ports_gdf['LATITUDE'] = ports_gdf['LATITUDE'].astype(float)
    ports_gdf['LONGITUDE'] = ports_gdf['LONGITUDE'].astype(float)

    # Create an empty list to store rows to keep
    keep_indices = []

    # Wrap the loop with tqdm for progress tracking
    for _, centroid_row in tqdm(centroids_gdf.iterrows(), total=len(centroids_gdf)):
        lon_centroid, lat_centroid = centroid_row['LONGITUDE'], centroid_row['LATITUDE']

        # Filter centroids within the specified range of lat and lon
        close_centroids_gdf = centroids_gdf[
            (abs(centroids_gdf['LONGITUDE'] - lon_centroid) < lon_threshold) &
            (abs(centroids_gdf['LATITUDE'] - lat_centroid) < lat_threshold)
        ]

        # Initialize a flag to keep track if the centroid is far from ports
        is_far = True

        for _, port_row in ports_gdf.iterrows():
            lon_port, lat_port = port_row['LONGITUDE'], port_row['LATITUDE']

            # Compute distance using the haversine_distance function
            distance = haversine_distance(lat_centroid, lon_centroid, lat_port, lon_port)

            # Check if the port is within the specified distance threshold
            if distance < distance_threshold:
                is_far = False
                break  # Stop checking further ports if one is within the threshold

        # Check if the centroid is far from ports
        if is_far:
            keep_indices.append(centroid_row.name)

    # Filter the centroids GeoDataFrame to keep only the far ones
    far_centroids_gdf = centroids_gdf.loc[keep_indices]

    return far_centroids_gdf
def clustering(stops,ports,points,e1=0.000035,n1=50,e2=0.0005,n2=1,file='scan.csv'):
    stops_clustered=dbscan(stops,e1,n1)
    stops_clustered=stops_clustered[stops_clustered['CLUSTER_LABEL']!=-1]
    centroids=get_cluster_centroids(stops_clustered,details=False)
    new_ports=from_pd_to_gpd(centroids_far_from_ports(centroids, ports))
    stops_clustered=dbscan(new_ports,e2,n2)
    stops_clustered=stops_clustered[stops_clustered['CLUSTER_LABEL']!=-1]
    centroids=get_cluster_centroids(stops_clustered,details=False)
    new_ports=from_pd_to_gpd(centroids_far_from_ports(centroids, ports))
    new_ports.to_csv(root+'output/Clustering'+file)
    perc=accuracy_NEW_PORTS(centroids, points,fast=True)
    return new_ports,perc

def dbscan_short(data, epsilon, min_samples):
    # Drop rows with non-numeric values in 'LONGITUDE' and 'LATITUDE' columns
    data = data.dropna(subset=['LONGITUDE', 'LATITUDE'], how='any', axis=0)

    # Check if the remaining values in 'LONGITUDE' and 'LATITUDE' columns are numeric
    numeric_check = pd.to_numeric(data['LONGITUDE'], errors='coerce').notna() & pd.to_numeric(data['LATITUDE'], errors='coerce').notna()

    # Filter out rows with non-numeric values in 'LONGITUDE' or 'LATITUDE'
    data = data[numeric_check]

    # Extract the coordinates for clustering
    coordinates = data[['LONGITUDE', 'LATITUDE']].values

    # Apply DBSCAN
    dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan_model.fit_predict(coordinates)

    # Add the cluster labels to the DataFrame
    data['Cluster'] = clusters

    return data

def doubble_DBSCAN(points, radius, n,ports, RADIUS=0.001, N=1):
    print('...clustering on stops...')
    stops_clustered = dbscan_short(points, radius, n) #first clustering
    stops_clustered=stops_clustered[stops_clustered['Cluster']!=-1] #remove not label clusters 
    centroids=get_cluster_centroids_short(stops_clustered,details=False) #get centroid 
    print('...clustering on centroids...')
    stops_clustered = dbscan_short(centroids, RADIUS, N) #selcond clustering, or aggregation
    stops_clustered=stops_clustered[stops_clustered['Cluster']!=-1] 
    centroids=get_cluster_centroids_short(stops_clustered,details=False)
    print('...removing redundant new ports...')
    new_ports=from_pd_to_gpd(centroids_far_from_ports_short(centroids, ports))# remove redundant ports
    denominator = len(centroids)
    TP = (denominator - len(new_ports)) / denominator if denominator != 0 else 0.0
    print('...removing offshore clusters...')
    new_ports=remove_offshore_ports(new_ports)
    return denominator,TP,new_ports


def country_code_2_to_3(country_code_2):
  """Converts a country code 2 to a country code 3.

  Args:
    country_code_2: A two-letter country code.

  Returns:
    A three-letter country code, or None if the country code 2 is not valid.
  """

  try:
    country = pycountry.countries.get(alpha_2=country_code_2)
    if country is not None:
      return country.alpha_3
  except KeyError:
    pass
  return None




def fill_missing_geocode(ports):
    """
    Fill missing 'PORT_NAME' and 'CNTR_CODE3' values in the 'ports' DataFrame using reverse geocoding.

    Args:
        ports (pd.DataFrame): The DataFrame containing 'PORT_NAME', 'CNTR_CODE3', 'LONGITUDE', and 'LATITUDE' columns.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled through reverse geocoding.
    """
    # Initialize the geocoder
    geolocator = rg.Geocoder()

    # Create a progress bar using tqdm
    for i in tqdm(range(len(ports)), desc="Geocoding Progress"):
        if pd.isna(ports['PORT_NAME'].iloc[i]):
            coordinate = (ports['LATITUDE'].iloc[i], ports['LONGITUDE'].iloc[i])
            location = geolocator.reverse(coordinate, exactly_one=True)  # Perform reverse geocoding
            if location:
                result = location.raw
                # Use your country_code_2_to_3 function to convert country code if needed
                ports.at[i, 'CNTR_CODE3'] = country_code_2_to_3(result['cc'])
                ports.at[i, 'PORT_NAME'] = country_code_2_to_3(result['name'])
    
    return ports
def fix_ports_df(ports_df):
    """Fixes a Pandas DataFrame containing port data by filling in missing PORT_NAME and CNTR_CODE3 values using reverse geocoding.

    Args:
        ports_df: A Pandas DataFrame containing port data.

    Returns:
        A Pandas DataFrame with the missing PORT_NAME and CNTR_CODE3 values filled in.
    """
    #read shp file to find the country for each port point
    country_gdf = gpd.read_file('G:/My Drive/Port Detection Algorithm/input/EEZ_Simplified.shp')
    # Filter out the rows with NA values in the PORT_NAME column.
    missing_data_indices = ports_df[(ports_df['PORT_NAME']=='NaN' )|(ports_df['PORT_CODE']=='NaN')|(ports_df['CNTR_CODE3']=='NaN')|(ports_df['CNTR_CODE3'].isna())].index
    # Iterate over the rows in the DataFrame.
    for i in tqdm(missing_data_indices):
        coordinate = (ports_df['LATITUDE'].iloc[i], ports_df['LONGITUDE'].iloc[i])
        result = rg.get(coordinate)  # Default mode = 2
        point = Point(coordinate[1], coordinate[0])  # Reverse the coordinates for Point
        country_code_3 = get_country_code(country_gdf, point)
        country_name = result['name'].upper()
        port_code = country_name[0:3].upper()

        # Set the CNTR_CODE3 and PORT_NAME columns based on the reverse geocoder result.
        ports_df.at[i, 'CNTR_CODE3'] = country_code_3
        ports_df.at[i, 'PORT_NAME'] = country_name
        ports_df.at[i, 'PORT_CODE'] = port_code

    return ports_df


def add_incremental_port_code(df):
    port_code_counts = {}
    new_port_codes = []

    for _, row in df.iterrows():
        port_code = row['PORT_CODE'].split("_")[0]

        if pd.notna(port_code):
            if port_code in port_code_counts:
                port_code_counts[port_code] += 1
                new_port_code = f'{port_code}_{port_code_counts[port_code]}'
            else:
                port_code_counts[port_code] = 1  # Start counting from 1 for a new port code
                new_port_code = port_code

            new_port_codes.append(new_port_code)
        else:
            # If PORT_CODE is NaN, keep it as is
            new_port_codes.append(port_code)

    df['PORT_CODE'] = new_port_codes
    return df


def add_incremental_port_name(df):
    port_code_counts = {}
    new_port_codes = []

    for i, row in df.iterrows():
        port_code = row['PORT_NAME']

        if pd.notna(port_code):
            if port_code in port_code_counts:
                port_code_counts[port_code] += 1
                new_port_code = f'{port_code}_{port_code_counts[port_code]}'
            else:
                port_code_counts[port_code] = 0
                new_port_code = port_code

            new_port_codes.append(new_port_code)
        else:
            # If PORT_CODE is NaN, keep it as is
            new_port_codes.append(port_code)

    df['PORT_NAME'] = new_port_codes
    return df

def haversine_distance_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * c
    return distance * 0.539957

def search_neighboors(PORT_CODE,ports):
  port=ports[ports['PORT_NAME']==PORT_CODE]
  lat = float(port['LATITUDE'].iloc[0] if isinstance(port['LATITUDE'], pd.Series) else float(port['LATITUDE']))
  lon = float(port['LONGITUDE'].iloc[0] if isinstance(port['LONGITUDE'], pd.Series) else float(port['LONGITUDE']))  
  lon_diff = abs((lon) - ports['LONGITUDE'])
  lat_diff = abs((lat) - ports['LATITUDE'])
  ports_filtered = ports[(lat_diff <= 0.2) | (lon_diff <= 0.2)]
  ports_filtered['DISTANCE'] = haversine_distance_vectorized((lon), (lat), ports_filtered['LONGITUDE'], ports_filtered['LATITUDE'])
  ports_filtered = ports_filtered[(ports_filtered['DISTANCE'] < 3 )& (ports_filtered['DISTANCE']!=0)]
  return list(ports_filtered.PORT_NAME)
# Import necessary libraries if not already imported

# Define the function to expand the cluster
def expand_cluster(seed_port, ports_df, visited_ports=None, max_depth=50):
    if visited_ports is None:
        visited_ports = set()

    # Check if we've reached the maximum recursion depth
    if max_depth <= 0:
        return visited_ports

    # Call the search_neighboors function for the seed port
    neighbors = search_neighboors(seed_port, ports_df)
    # Add the seed port to the visited_ports set
    visited_ports.add(seed_port)

    # Recursively explore neighbors
    for neighbor in neighbors:
        if neighbor not in visited_ports:
            visited_ports.update(expand_cluster(neighbor, ports_df, visited_ports, max_depth - 1))

    return list(visited_ports)


def replace_line_manager(row):
    if row['Cluster'] == 0:
        return row['PORT_NAME']
    
    
    #  Port_Groups_foranalysis
def export_Port_Groups_foranalysis(ports,path='C:/Users/TE/Documents/Jacopo Nudo/Transport & Environment/git_repo/VCC_SEA_models/SVA/databases/inputs/geographics/PORTS/Port_Groups_foranalysis.xlsx'):
    line_manager_table = {}
    for lm in ports['Line_manager'].unique():
        port_codes = list(ports[ports['Line_manager'] == lm]['PORT_NAME'])
        if len(port_codes) > 1:
            line_manager_table[lm] = port_codes

    df = pd.DataFrame(line_manager_table.items(), columns=['Key', 'Values'])

    # Split the 'Values' column into separate columns
    df = df['Values'].apply(pd.Series).rename(lambda x: f'Value_{x + 1}', axis=1)

    # Rename the 'Key' column to match your desired output
    df = df.rename(columns={'Key': 'Line_manager'})

    # Fill NaN values with appropriate placeholders
    df = df.fillna('')

    # Rename columns if necessary
    df.columns = [
    'Part0', 'Part1', 'Part2', 'Part3', 'Part4', 'Part5', 'Part6', 'Part7',
    'Part8', 'Part9', 'Part10', 'Part11', 'Part12', 'Part13', 'Part14', 'Part15',
    'Part16', 'Part17', 'Part18','Part19', 'Part20', 'Part21','Part22', 'Part23',
    'Part24', 'Part25', 'Part26','Part27']

    # Export the DataFrame to an Excel file
    df.to_excel(path, index=False)


def stops_partition_on_ports(root):
    stops = pd.read_csv(root + 'input/stops_all.csv')
    ports = pd.read_csv(root + 'output/PORTS/PORTS.csv')
    vessels = pd.read_excel(root + 'input/db_vcc_final_seamodel.xlsx')
    # Merge stops with vessel data
    stops = stops.merge(vessels[['imo', 'gt', 'ship_class']], left_on='imo', right_on='imo', how='left')

    # Read and process hierarchy data
    hierarchy = pd.read_csv(root + 'input/well_known_ports.csv')
    hierarchy.dropna(inplace=True)
    well_known = hierarchy.values.flatten().tolist()

    # Convert columns to numeric
    ports[['LATITUDE', 'LONGITUDE']] = ports[['LATITUDE', 'LONGITUDE']].apply(pd.to_numeric, errors='coerce')
    stops[['LATITUDE', 'LONGITUDE']] = stops[['LATITUDE', 'LONGITUDE']].apply(pd.to_numeric, errors='coerce')

    # Initialize columns for ports assigned and distance
    stops['PORTS_ASSIGNED'] = ""
    stops['PORTS_DISTANCE'] = 0.0

    # Iterate through stops to assign ports
    for _, stop in tqdm(stops.iterrows(), total=stops.shape[0]):
        lon = stop["LONGITUDE"]
        lat = stop['LATITUDE']

        # Filter ports within specified range
        ports_filtered = ports[(abs(lat - ports['LATITUDE']) <= 0.3) & (abs(lon - ports['LONGITUDE']) <= 0.9)]

        if not ports_filtered.empty:
            # Calculate distance and assign closest port
            ports_filtered.loc[:, 'DISTANCE'] = haversine_vectorized(lon, lat, ports_filtered['LONGITUDE'], ports_filtered['LATITUDE'])
            min_distance = ports_filtered['DISTANCE'].min()

            if min_distance < 10:
                closest_ports = ports_filtered[ports_filtered['DISTANCE'] == min_distance]['PORT_CODE']
                stops.at[_, 'PORTS_ASSIGNED'] = str(closest_ports.values[0])
                stops.at[_, 'PORTS_DISTANCE'] = min_distance

    # Save processed data
    stops.to_csv(root + 'outputs/Partitions/stops_partition.csv')

from shapely.geometry import Point
import geopandas as gpd

def get_country_code(gdf, point):
    """
    Get the country code for a given point within a GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing country polygons with a 'CNTR_CODE3' column.
    - point (Point): Shapely Point geometry representing the location.

    Returns:
    - str: Country code for the given point, or 'NO COUNTRY FOUND' if not found or not unique.
    """
    # Create a spatial index for the GeoDataFrame
    spatial_index = gdf.sindex
    
    # Use the spatial index to filter potential candidates based on intersection
    potential_candidates = list(spatial_index.intersection(point.bounds))
    
    # Filter potential candidates based on exact intersection with the point
    candidates = gdf.loc[potential_candidates][gdf.geometry.contains(point)]
    
    if len(candidates) == 0:
        print(f"No country found for point at coordinates {point.x}, {point.y}")
        return 'NO COUNTRY FOUND'
    elif len(candidates) == 1:
        selected_country = candidates['CNTR_CODE3'].values[0]
        print(f"Point at coordinates {point.x}, {point.y} is in {selected_country}")
        return selected_country
    else:
        # Calculate distances from the point to the borders of candidate polygons
        distances = candidates.geometry.apply(lambda geom: point.distance(geom.boundary))
        
        # Find the index of the farthest polygon
        farthest_index = distances.idxmax()
        
        # Return the country code of the farthest polygon
        selected_country = gdf.loc[farthest_index, 'CNTR_CODE3']
        print(f"Point at coordinates {point.x}, {point.y} is in {selected_country}, instead of {candidates['CNTR_CODE3'].values[1]}")
        return selected_country

def remove_offshore_ports(new_ports):
    countries_path = "C:/Users/TE/Documents/git_repo/VCC_SEA_models/SVA/databases/inputs/geographics/COUNTRIES/COUNTRIES_SIMPLIFIED.shp"
    countries_gdf = gpd.read_file(countries_path)

    # Use tqdm to track progress
    for index, row in tqdm(new_ports.iterrows(), total=len(new_ports), desc="Removing off shore ports poiports"):
        given_point = Point(row['LONGITUDE'], row['LATITUDE'])
        closest_country_index = countries_gdf.geometry.apply(lambda geom: geom.distance(given_point)).idxmin()
        closest_country = countries_gdf.loc[closest_country_index, 'geometry']
    
        # Calculate the distance from the land (closest polygon)
        distance_to_land = given_point.distance(closest_country)
        if distance_to_land > 0:
            new_ports = new_ports.drop(index)

    return new_ports





#da testare sui voyages V1 se questi porti nuovi coprono piu new port di quelli attualmente su git. 
from pyproj import Proj, transform
import pandas as pd

def project_to_EPSG4326(df):
    # Definisci il sistema di riferimento di partenza (WGS84)
    src_crs = Proj(init='epsg:4326')

    # Definisci il sistema di riferimento di destinazione (EPSG4326)
    dst_crs = Proj(init='epsg:4326')

    # Effettua la trasformazione delle coordinate per ciascuna riga del DataFrame
    new_longitudes = []
    new_latitudes = []

    for index, row in df.iterrows():
        longitude = row['longitude']
        latitude = row['latitude']
        new_longitude, new_latitude = transform(src_crs, dst_crs, longitude, latitude)
        new_longitudes.append(new_longitude)
        new_latitudes.append(new_latitude)

    # Crea un nuovo DataFrame con le coordinate trasformate
    transformed_df = pd.DataFrame({
        'longitude': new_longitudes,
        'latitude': new_latitudes
    })

    return transformed_df

# Esempio di utilizzo:
# Supponiamo che 'df' sia il tuo DataFrame con colonne 'longitude' e 'latitude'
# df_transformed = project_to_EPSG4326(df)
# print(df_transformed)
