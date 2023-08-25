# this is for leave-one-out scheme
import os
import random
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from data.datasets import AMIGOS, series_collate
from architecture.MainNetwork import MainNetwork
from utils import *

# define configurations
torch.backends.cudnn.enabled = False
loader_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True, 'drop_last': True}
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = 'data/face_segments'
labels_path = 'data/Data_Preprocessed_segmented.json'
vids_dir = 'data/vids_segments'
remove_mov = 'data/ignore_mov.json'
num_class = 4096
batch_size = 4
learning_rate = 1e-05
epochs = 15
alpha = 1
beta = 1
scale_factor = 1
gamma = scale_factor
downsample = 8
normalize_val = {
    'AR': {'min': -0.42818420244970845, 'range': 0.40530026133943436 - -0.42818420244970845},
    'ECG': {'min': -2281.0594032292756, 'range': 2340.911172156569 - -2281.0594032292756},
}

# define folder path for logging
savemodel = '/scratch/ec22150/cccN/models/'
if not os.path.exists(savemodel):
    os.makedirs(savemodel)
log_dir = '/scratch/ec22150/cccN/log'
log_writer = SummaryWriter(os.path.join(*[log_dir, 'AMIGOS', 'Test', datetime.now().strftime('%b%d_%H-%M-%S_eval')]))

# construct dataset
x_transform = transforms.Compose([
    transforms.Normalize([0.4168, 0.3074, 0.2607], [0.2426, 0.1997, 0.1870])
])

y_transform = transforms.Compose([
    torch.FloatTensor
])

val_dataset = AMIGOS(
    root_path=root_path,
    labels_path=labels_path,
    vids_dir=vids_dir,
    remove_mov=remove_mov,
    x_transform=x_transform,
    y_transform=y_transform,
    downsample=downsample,
    normalize_val=normalize_val,
    # normalize=False
)

# get model
model_lst = os.listdir(savemodel)
model_lst = [os.path.join(savemodel, fname) for fname in model_lst]
model = MainNetwork(num_class).to(device)

y_pred_ARs = []
y_true_ARs = []
y_pred_ECGs = []
y_true_ECGs = []
for model_path in model_lst:
    print('mm')
    test_loader = eval_dataloader(model_path, val_dataset, batch_size, loader_kwargs)
    
    if len(test_loader) == 0:
        continue
    state_dicts = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(state_dicts['model'])

    y_pred_AR, y_true_AR, y_pred_ECG, y_true_ECG = run_val(model, test_loader, batch_size)
    y_pred_ARs.append(y_pred_AR.cpu())
    y_true_ARs.append(y_true_AR.cpu())
    y_pred_ECGs.append(y_pred_ECG.cpu())
    y_true_ECGs.append(y_true_ECG.cpu())
    torch.cuda.empty_cache()

# del model
print('d1')
y_pred_ARs = torch.cat(y_pred_ARs)
y_true_ARs = torch.cat(y_true_ARs)
y_pred_ECGs = torch.cat(y_pred_ECGs)
y_true_ECGs = torch.cat(y_true_ECGs)
print('d2')

val_log(log_writer, alpha, scale_factor, y_pred_ARs, y_true_ARs, y_pred_ECGs, y_true_ECGs, val=True)
print('d2')

log_writer.flush()
print('d2')
log_writer.close()
print('d2')
