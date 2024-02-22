df

###function




###
from math import radians, sin, cos, sqrt, atan2

def haversine(latlon1, latlon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    lat1, lon1 = radians(latlon1[0]), radians(latlon1[1])
    lat2, lon2 = radians(latlon2[0]), radians(latlon2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance



df=df[0:1000]
from scipy.spatial.distance import pdist, squareform
distance_matrix = squareform(pdist(df, (lambda u,v: haversine(u,v))))





import numpy as np
from sklearn.cluster import DBSCAN

# Assuming 'df' is your DataFrame containing latitude and longitude columns
# Convert latitude and longitude columns to radians before passing to DBSCAN
df_radians = np.radians(df[['latitude', 'longitude']])

# Create DBSCAN object with haversine metric
db = DBSCAN(eps=2/6371., min_samples=5, algorithm='ball_tree', metric='haversine').fit(df_radians)

