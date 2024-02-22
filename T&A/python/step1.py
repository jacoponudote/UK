import os
import pandas as pd
from tqdm import tqdm
import random

def filter_uk_coordinates(df):
    uk_longitude_min = -10
    uk_longitude_max = 2
    uk_latitude_min = 49
    uk_latitude_max = 61

    # Condizioni per il filtraggio
    uk_data = df[
        (df['longitude'].between(uk_longitude_min, uk_longitude_max, inclusive=True)) &
        (df['latitude'].between(uk_latitude_min, uk_latitude_max, inclusive=True)) &
        ((df['sog'] < 3) | (df['navigational_status'].str.contains('Moored')))
    ]

    return uk_data

# Function to process a single file
def process_file(file_path):
    try:
        # Read CSV file into a DataFrame, only reading the necessary columns
        df = pd.read_csv(file_path, usecols=['imo', 'longitude', 'latitude','navigational_status','sog'])

        # Filter the DataFrame based on UK coordinates
        uk_data = filter_uk_coordinates(df)

        # If there is at least one row, return the IMO code
        if not uk_data.empty:
            return uk_data['imo'].iloc[0]
        else:
            return None
    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: No columns to parse from file: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Folder path containing CSV files
folder_path = "G:/.shortcut-targets-by-id/1P_4N7G6u0dl2LSK74MfP-JMcjCV3r-L0/AIS/Shipping_AIS_Spire_hourly/2022/"

# Get the list of CSV files in the folder
files_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Shuffle the list of files
random.shuffle(files_list)

# List to store IMO codes
imo_codes_with_data = []

# List to store file paths
file_paths = []

# Process each file with tqdm progress bar
for file_path in tqdm(files_list, desc="Processing files"):
    imo_code = process_file(file_path)
    if imo_code:
        imo_codes_with_data.append(imo_code)
        file_paths.append(file_path)

# Remove duplicate IMO codes and print the list
imo_codes_with_data = list(set(imo_codes_with_data))
print("IMO codes with at least one row within the UK coordinates:")
print(imo_codes_with_data)

# Define the file path to save the list of IMO codes
output_file_path = os.path.join("G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis", "imo_codes_2.txt")

# Save the list of IMO codes to the file
with open(output_file_path, 'w') as file:
    for imo_code in imo_codes_with_data:
        file.write(str(imo_code) + '\n')

print("IMO codes have been saved to:", output_file_path)

# Create DataFrame with IMO codes
df = pd.DataFrame({
    'IMO_Code': imo_codes_with_data,
    'File_Path': file_paths
})

# Define the file path to save the DataFrame
output_df_file_path = os.path.join(os.path.dirname(output_file_path), 'imos_UK_port_Ranking.csv')

# Save the DataFrame to a CSV file
df.to_csv(output_df_file_path, index=False)
print("DataFrame has been saved to:", output_df_file_path)
