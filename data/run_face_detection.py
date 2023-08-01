import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_path = 'data/vids_segments'
dst_dir = 'data/face_segments'

root_path = '/home/ec22150/MSCPRJ/MSC-PRJ/new2'
dst_dir = '/home/ec22150/MSCPRJ/MSC-PRJ/new3'

from AMIGOS import face_detection_fm
face_detection_fm(root_path, dst_dir, device)

# processing too slow
# from AMIGOS import face_detection
# face_detection(root_path, dst_dir)