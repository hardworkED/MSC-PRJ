"""
Adapted from https://github.com/NickyFot/ACMMM22_LearningLabelRelationships
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.nn.utils import rnn
import torchvision.transforms as transforms
from data.AMIGOS import ignore_mov

def series_collate(batch):
    x = [item[0] for item in batch]
    y1 = [item[1] for item in batch]
    y2 = [item[2] for item in batch]
    idx = [item[3] for item in batch]
    y1 = torch.stack(y1, 0)
    y2 = torch.stack(y2, 0)
    x = rnn.pad_sequence(x, batch_first=True)
    # x = torch.stack(x)
    return x, y1, y2, idx

class AMIGOS(data.Dataset):
    """
    Class to handle AMIGOS Dataset.
    """
    def __init__(self, root_path, labels_path, vids_dir, x_transform, y_transform, downsample=5, remove_mov=None):
        """
        Dataset constructor
        :param root_path: (str) path to root of face segments
        :param labels_path: (str) path to preprocessed data json file
        :param vids_dir: (str) path to root of video segments
        :param x_transform: (callable) transformation to apply to segment of frames
        :param y_transform: (callable) transformation to apply to labels
        :returns AMIGOS Dataset object
        """
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.data = self.make_dataset(root_path, labels_path, vids_dir, remove_mov)
        self.labels = [[self.data[uid][vid][seg_id]['AR'], self.data[uid][vid][seg_id]['ECG']] for uid in self.data for vid in self.data[uid] for seg_id in self.data[uid][vid]]
        self.idxs = [(i, uid, vid, seg_id) for uid in self.data.keys() for vid in self.data[uid].keys() for i, seg_id in enumerate(self.data[uid][vid].keys())]
        self.indices = list(range(0, len(self.idxs)))
        self.downsample = downsample

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.labels = [[self.data[uid][vid][seg_id]['AR'], self.data[uid][vid][seg_id]['ECG']] for uid in self.data for vid in self.data[uid] for seg_id in self.data[uid][vid]]
        self.idxs = [(i, uid, vid, seg_id) for uid in self.data.keys() for vid in self.data[uid].keys() for i, seg_id in enumerate(self.data[uid][vid].keys())]
        self.indices = list(range(0, len(self.idxs)))

    def __getitem__(self, index):
        index = self.idxs[index]
        
        path = self.data[index[1]][index[2]][index[3]]['frames_path']
        x = self.loader(path)
        x = torch.stack(x, 0)
        if self.x_transform:
            x = self.x_transform(x)
        ys = [self.data[index[1]][index[2]][index[3]]['AR'], self.data[index[1]][index[2]][index[3]]['ECG']]
        if self.y_transform:
            ys = [self.y_transform(y) for y in ys]
        return x, ys[0], ys[1], index

    def __len__(self):
        return len(self.data)
    
    def loader(self, path):
        frames = [f for f in os.listdir(path) if '.jpg' in f]
        frames.sort()
        # downsampling
        frames = [frames[idi] for idi in range(len(frames)) if (idi % self.downsample) == 0]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        ret = [transform(cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB)) for f in frames]
        return ret
    
    def make_dataset(self, root_path, labels_path, vids_dir, remove_mov):
        # filter out videos that did not meet the requirements
        # when vids_dir is removed to free out disk space
        if remove_mov is None or not os.path.exists(remove_mov):
            to_remove_mov = ignore_mov(vids_dir, root_path)
            to_remove_mov += ['P11_18_face', 'P13_58_face', 'P39_10_face', 'P8', 'P24', 'P28']
            if remove_mov:
                dt = {'remove_mov': to_remove_mov}
                with open(remove_mov, 'w') as f:
                    json.dump(dt, f)
        else:
            with open(remove_mov, 'r') as f:
                to_remove_mov = json.load(f)['remove_mov']
        segment_paths = []
        for class_name in os.listdir(root_path):
            if '_face' in class_name:
                class_path = os.path.join(root_path, class_name)
                for filename in os.listdir(class_path):
                    if '_face' in filename:
                        segment_path = os.path.join(class_path, filename)
                        if not any([remove in segment_path for remove in to_remove_mov]):
                            segment_paths.append(segment_path)

        with open(labels_path, 'r') as f:
            data_preprocessed = json.load(f)

        dt = {}
        for segment_path in segment_paths:
            vid_name = os.path.basename(segment_path).split('_')
            uid = int(vid_name[0][1:])
            if uid not in dt.keys():
                dt[uid] = {}
            vid = vid_name[1]
            dt[uid][vid] = {segment: {
                'frames_path': os.path.join(segment_path, str(segment)),
                'AR': np.asarray([data_preprocessed['Data_Preprocessed_P{:02d}'.format(uid)][vid]['AR'][segment]['arousal'], data_preprocessed['Data_Preprocessed_P{:02d}'.format(uid)][vid]['AR'][segment]['valence']]),
                'ECG': np.asarray([np.asarray(data_preprocessed['Data_Preprocessed_P{:02d}'.format(uid)][vid]['ECG_L'][segment]), np.asarray(data_preprocessed['Data_Preprocessed_P{:02d}'.format(uid)][vid]['ECG_R'][segment])])
            } for segment in data_preprocessed['Data_Preprocessed_P{:02d}'.format(uid)][vid]['AR'].keys()}
            # exclude video if there is nan value in ECG
            ECGs = [dt[uid][vid][seg]['ECG'] for seg in dt[uid][vid].keys()]
            ECGs = np.array(ECGs).ravel()
            if any(np.isnan(ECGs)):
                del dt[uid][vid]
        return dt