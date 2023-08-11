import pandas as pd
import os
import glob

# set files path
path_folder = 'path_to_folder_with_samples'
name_output_csv_merged_file = 'merged_samples.csv'
paths_to_csv_files = glob.glob(f'{path_folder}/*.csv')
paths_to_csv_files.sort()
print(f"There are {len(paths_to_csv_files)} csv file to be merged.")

for csv_file in paths_to_csv_files:
    if os.stat(csv_file).st_size < 5:
        os.remove(csv_file)

paths_to_csv_files = glob.glob(f'{path_folder}/*.csv')
paths_to_csv_files.sort()
print(f"There are {len(paths_to_csv_files)} csv file to be merged after removing empty ones.")

# merge files
dataFrame = pd.concat(map(pd.read_csv, paths_to_csv_files), ignore_index=True)

pd.DataFrame.to_csv(dataFrame, os.path.join(path_folder, name_output_csv_merged_file), index=False)
print('Done!')