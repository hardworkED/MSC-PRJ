import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.chdir('/home/ec22150/MSCPRJ/MSC-PRJ')

root_path = 'data/vids_segments'
dst_dir = 'data/face_segments'

from AMIGOS import face_detection_fm
face_detection_fm(root_path, dst_dir, device, 'Exp1_P2')

# processing too slow
# from AMIGOS import face_detection
# face_detection(root_path, dst_dir)