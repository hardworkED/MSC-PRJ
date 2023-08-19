import os
import random
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from data.datasets import AMIGOS, series_collate
from architecture.MainNetwork import MainNetwork
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = False
loader_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True, 'drop_last': True}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = 'data/face_segments'
labels_path = 'data/Data_Preprocessed_segmented.json'
vids_dir = 'data/vids_segments'
remove_mov = 'data/ignore_mov.json'
num_class = 4096
batch_size = 4
learning_rate = 1e-05
epochs = 12
alpha = 2
beta = 1
scale_factor = 1
gamma = scale_factor
downsample = 8
normalize_val = {
    'AR': {'min': -0.42818420244970845, 'range': 0.40530026133943436 - -0.42818420244970845},
    'ECG': {'min': -2281.0594032292756, 'range': 2340.911172156569 - -2281.0594032292756},
}

x_transform = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.4168, 0.3074, 0.2607], [0.2426, 0.1997, 0.1870])
])

y_transform = transforms.Compose([
    torch.FloatTensor
])

train_dataset = AMIGOS(
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

x_transform = transforms.Compose([
    transforms.Normalize([0.4168, 0.3074, 0.2607], [0.2426, 0.1997, 0.1870])
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

mode = 'rmse'
savemodel = 'models/' + mode
if not os.path.exists(savemodel):
    os.makedirs(savemodel)
log_dir = 'log/' + mode
log_writer = SummaryWriter(os.path.join(*[log_dir, 'AMIGOS', 'Train', datetime.now().strftime('%b%d_%H-%M-%S')]))

output_names = ['AR', 'ECG']
idxs = [uid for uid in train_dataset.data.keys()]
idxs.sort()
train_uids = idxs[:(len(idxs) // 5) * 4]
# train_uids = random.sample(idxs, (len(idxs) // 5) * 4)
# select 20% samples from each training uid
actual_train = []
for tidx in train_uids:
    samples = [idx[0] for idx in train_dataset.idxs if idx[1] == tidx]
    actual_train.extend(random.sample(samples, len(samples) // 50))
random.shuffle(actual_train)
val_idx = [idx[0] for idx in train_dataset.idxs if idx[1] not in train_uids][:10]
print('Training UIDs {} with {} samples'.format(train_uids, len(actual_train)))

train_set = data.Subset(train_dataset, actual_train)
val_set = data.Subset(val_dataset, val_idx)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    collate_fn=series_collate,
    **loader_kwargs
)
test_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    collate_fn=series_collate,
    **loader_kwargs
)

# training
lr = learning_rate
model = MainNetwork(num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
scaler = torch.cuda.amp.GradScaler()
loss_hist = []

iter_idx = 0
for epoch in range(epochs):
    model.train()
    if (epoch % 5 == 0) and (epoch != 0):
        lr *= 0.1
        optimizer.param_groups[0]['lr'] = lr
    num_iter = len(train_loader)
    for batch_idx, (inputs, labels1, labels2, _) in enumerate(train_loader):
        iter_idx = (epoch * num_iter) + batch_idx
        inputs, labels1, labels2 = inputs.cuda(), labels1.cuda(), labels2.cuda()
        losses = []
        optimizer.zero_grad()
        with torch.autocast(device.type):
            outputs = model(inputs)
            labels = [labels1, labels2]
            for i in range(len(labels)):
                if output_names[i] == 'ECG':
                    mae, mse, rmse, pcc, ccc = eval_metrics(outputs[i].permute(0, 2, 1).reshape((batch_size * 2560, 2)), labels[i].permute(0, 2, 1).reshape((batch_size * 2560, 2)))
                    # loss = (1-ccc).mean() + 2 * relational_loss(outputs[i].permute(0, 2, 1).reshape((batch_size * 2560, 2)), labels[i].permute(0, 2, 1).reshape((batch_size * 2560, 2)))
                else:
                    mae, mse, rmse, pcc, ccc = eval_metrics(outputs[i], labels[i])
                    # loss = (1-ccc).mean() + 2 * relational_loss(outputs[i], labels[i])
                # calculating loss
                # loss = (1-ccc).mean() + alpha * rmse
                # loss = rmse
                # loss = (1-ccc).mean()
                loss = (1-ccc).mean()
                losses.append(loss)

                logging('Train', output_names[i], log_writer, loss, mae, mse, rmse, pcc, ccc, iter_idx)
                print('\n Epoch [{}/{}] Iter[{}/{}] Mode: {} \t loss: {:.2f} \t MAE: {:.2f} \t MSE: {:.2f} \t RMSE: {:.2f} \t PCC:{} \t CCC:{} '.format(
                    epoch,
                    epochs,
                    batch_idx + 1,
                    num_iter,
                    output_names[i],
                    loss.item(),
                    mae.item(),
                    mse.item(),
                    rmse.item(),
                    ['%.2f' % elem for elem in pcc.tolist()],
                    ['%.2f' % elem for elem in ccc.tolist()],
                ))

        loss = beta * losses[0] + gamma * losses[1]
        log_writer.add_scalar('TotalLoss/{}'.format('Train'), loss, iter_idx)
        print('\n Epoch [{}/{}] Iter[{}/{}]\t TotalLoss: {:.2f}'.format(
            epoch,
            epochs,
            batch_idx + 1,
            num_iter,
            loss.item()
        ))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # release GPU memory
        # del inputs, labels, labels1, labels2, outputs
        # torch.cuda.empty_cache()
    
    # validation
    loss = run_val2(model, test_loader, batch_size, epoch, log_writer, alpha, val=True)
    print('\n Epoch [{}/{}] \t TotalLoss: {:.2f}'.format(
        epoch,
        epochs,
        loss.item()
    ))
    log_writer.flush()
    print('\n')
    # del losses, y_pred_AR, y_true_AR, y_pred_ECG, y_true_ECG
    # torch.cuda.empty_cache()
    loss_hist.append(loss)
    if loss_hist[-1] == min(loss_hist):
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
        },
            os.path.join(savemodel, 'epoch_{}.pth.tar'.format(epoch))
        )
    log_writer.flush()
    
del model
log_writer.close()