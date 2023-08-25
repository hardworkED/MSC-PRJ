from AMIGOS import data_preprocessed, ecg_downsampling, ecg_segmentation
import os

data_dir = 'data/'
data_preprocessed_path = os.path.join(data_dir, 'Data_Preprocessed')
json_filename = 'Data_Preprocessed.json'

# preprocessed labels from .mat to .json
# data_preprocessed(data_dir, data_preprocessed_path, json_filename)

file_path = os.path.join(data_dir, json_filename)

# if downsampling is reuquired
# ecg_downsampling(file_path, new_freq=25)

# segment ecg data according to video segments
ecg_segmentation(file_path)