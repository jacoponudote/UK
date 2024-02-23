import sys
folder_path = r'G:/Other computers/My PC/uk_port_ranking/'
sys.path.append('C:/Users/TE/Documents/UK/T&A/python/')
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm 
from UK_Package.functions import *

port_co2_dict = {}

# Carica il file dei porti
ports_path = r'C:/Users/TE/Documents/git_repo/VCC_SEA_models/SVA/databases/inputs/geographics/PORTS/PORTS.csv'
ports_df = pd.read_csv(ports_path)

# Crea la geometria per i porti
ports_geometry = [Point(xy) for xy in zip(ports_df['LONGITUDE'], ports_df['LATITUDE'])]
ports_gdf = gpd.GeoDataFrame(ports_df, geometry=ports_geometry, crs="EPSG:4326")

# Aggiungi un buffer ai porti
ports_gdf['geometry'] = ports_gdf['geometry'].buffer(0.8)  # Impostiamo 0.001 come distanza del buffer (da adattare)

# Read the file into a DataFrame
columns_needed = ['DIST_SGNL', 'PORT_STOP', 'LONGITUDE', 'LATITUDE', 'E_CO2_kg']
files = [filename for filename in os.listdir(folder_path) if "hourly" in filename and filename.endswith(".csv")]



for filename in tqdm(files):
    if "hourly" in filename and filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
    
        # Read only the specified columns from the CSV file
        signals_df = pd.read_csv(file_path, usecols=columns_needed, low_memory=False)
        signals_df = signals_df[(signals_df['DIST_SGNL'] < 3) & ~signals_df['PORT_STOP'].isna()]
        if len(signals_df)==0:
            continue
        
        # Crea la geometria per i segnali
        signals_geometry = [Point(xy) for xy in zip(signals_df['LONGITUDE'], signals_df['LATITUDE'])]
        signals_gdf = gpd.GeoDataFrame(signals_df, geometry=signals_geometry, crs="EPSG:4326")
        
        # Esegui l'operazione di join spaziale
        joined = gpd.sjoin(signals_gdf, ports_gdf, how="inner", op="intersects")
        joined['Distance_to_Port'] = joined.apply( lambda row: haversine_distance( row['LONGITUDE_left'],row['LATITUDE_left'], row['LONGITUDE_right'],row['LATITUDE_right']),axis=1)
        joined=joined[joined['Distance_to_Port']<6]
        joined=joined[['E_CO2_kg','PORT_NAME']]
        aggregated_data = joined.groupby('PORT_NAME')['E_CO2_kg'].sum().reset_index()
        
        for index, row in aggregated_data.iterrows():
            port_name = row['PORT_NAME']
            co2_value = row['E_CO2_kg']
            
            # Check if the port_name already exists in the dictionary
            if port_name in port_co2_dict:
                # If the port_name exists, sum the CO2 value
                port_co2_dict[port_name] += co2_value
            else:
                # If the port_name does not exist, add it to the dictionary
                port_co2_dict[port_name] = co2_value
        

pd.DataFrame(list(port_co2_dict.items()), columns=['PORT_NAME', 'E_CO2_kg']).to_csv('C:/Users/TE/Documents/UK/T&A/stored_data/emissions_in port_6nm')


