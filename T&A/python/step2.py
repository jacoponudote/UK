
root = 'G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/'
import pandas as pd
from tqdm import tqdm
import os
import csv


#Structure
STOP_DETECTION=False
STOPS_READY=True



#Parameters
radius=0.00002 #200meters
n=2 #minimum number of vessels to detect a branch.
radius_2=0.002 #20km
n_2=1 #minimum number of branch to detect a port


#Warring
#Insert the version of the Port DB to update here Port Detection Algorithm/input/PORTS/PORTS.csv
#The output will be in the twin folder but in Port Detection Algorithm/output/PORTS/PORTS.csv

#Multiprocess 
n_threads=4 # set to 0 to avoid multiptocessing
thread=3

if STOP_DETECTION:
    # Insert folder with AIS for which are you looking for PORT DETECTION
    print(" PART 0: Detects stops from ais files")
    all_paths_df = pd.read_csv('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/imos_UK_port_Ranking.csv')#,nrows=5000) # to run on a sample 
    batch_size = 50  #SET number of vessels to read for each iteration
    n=int(len(all_paths_df)/n_threads)
    for i in tqdm(range(n*thread,(n)*(thread+1), batch_size)):
        start_index = i
        stop_index = min(i + batch_size,(n)*(thread+1))  
        stops = process_files('G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/imos_UK_port_Ranking.csv', start_index,stop_index , stops_detection, f'Stops_{start_index}')
        
if STOPS_READY:  
    print(" PART 1: Merge all stops")
    folder_path = 'G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/stops/'#concatenate all the stop files from here 
    output_file = 'G:/.shortcut-targets-by-id/1_zWaxCE-0eNdkvvp0stjhg7KWcSUSHEI/202402_UK_port_ranking/T&A/raw_data/for_analysis/stops_all.csv'#to here
    file_list = os.listdir(folder_path)
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                process_csv_file(file_path, writer)
            except Exception as e:
                warnings.warn(f"Error processing file '{file_name}': {str(e)}", UserWarning)
                