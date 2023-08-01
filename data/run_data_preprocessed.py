from AMIGOS import data_preprocessed, ecg_downsampling, ecg_segmentation
import os

data_dir = 'data/'
data_preprocessed_path = os.path.join(data_dir, 'Data_Preprocessed')
json_filename = 'Data_Preprocessed.json'

# data_preprocessed(data_dir, data_preprocessed_path, json_filename)

file_path = os.path.join(data_dir, json_filename)
# ecg_downsampling(file_path, new_freq=25)

ecg_segmentation(file_path)