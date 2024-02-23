import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Carica il file dei segnali
signals_path = r'G:/Other computers/My PC/uk_port_ranking/2022_6409351_emissions_hourly.csv'
signals_df = pd.read_csv(signals_path)

# Carica il file dei porti
ports_path = r'C:/Users/TE/Documents/git_repo/VCC_SEA_models/SVA/databases/inputs/geographics/PORTS/PORTS.csv'
ports_df = pd.read_csv(ports_path)

# Crea la geometria per i segnali
signals_geometry = [Point(xy) for xy in zip(signals_df['LONGITUDE'], signals_df['LATITUDE'])]
signals_gdf = gpd.GeoDataFrame(signals_df, geometry=signals_geometry, crs="EPSG:4326")

# Crea la geometria per i porti
ports_geometry = [Point(xy) for xy in zip(ports_df['LONGITUDE'], ports_df['LATITUDE'])]
ports_gdf = gpd.GeoDataFrame(ports_df, geometry=ports_geometry, crs="EPSG:4326")

# Aggiungi un buffer ai porti
ports_gdf['geometry'] = ports_gdf['geometry'].buffer(0.12)  # Impostiamo 0.001 come distanza del buffer (da adattare)

# Esegui l'operazione di join spaziale
joined = gpd.sjoin(signals_gdf, ports_gdf, how="inner", op="intersects")



joined['Distance_to_Port'] = joined.apply(
    lambda row: haversine_distance(
        row['LONGITUDE_left'],
        row['LATITUDE_left'],
        row['LONGITUDE_right'],
        row['LATITUDE_right']
    ),
    axis=1
)


joined['Distance_to_Port']