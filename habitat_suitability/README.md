# Wetlands habitat suitability 

## Overview
We estimated wetlands habitat suitability by training a Random Forest model on samples from the CCAP dataset (class 13) as ground truth for coastal wetlands forests.

Steps to follow to replicate this work:
1. Create a dataset following the example from the notebook: ```model_utilities/wetlands_habitat_suitability_dataset_creation.ipynb```
NOTE: you need to create a gridded shapefile to use this notebook (use QGIS for this and then import the shapefile in GEE as an asset)
2. All samples files will be stored in GCP buckets: download them and merge all of them with the script ```model_utilities/merge_csv_files.py```
3. Use the merged csv file for training a Random Forest model following the example of te notebook: ```model_utilities/model_builder_wetlands_lab.ipynb```
4. Inference on tiles is also ran in the notebook ```model_utilities/model_builder_wetlands_lab.ipynb```


