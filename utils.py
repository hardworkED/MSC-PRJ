"""
Adapted from https://github.com/NickyFot/ACMMM22_LearningLabelRelationships
"""

import os
from data.datasets import series_collate
from torch.utils import data
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt


def PCC(a: torch.tensor, b: torch.tensor):
    am = torch.mean(a, dim=0)
    bm = torch.mean(b, dim=0)
    num = torch.sum((a - am) * (b - bm), dim=0)
    den = torch.sqrt(sum((a - am) ** 2) * sum((b - bm) ** 2)) + 1e-5
    return num/den


def CCC(a: torch.tensor, b: torch.tensor):
    rho = 2 * PCC(a,b) * a.std(dim=0, unbiased=False) * b.std(dim=0, unbiased=False)
    rho /= (a.var(dim=0, unbiased=False) + b.var(dim=0, unbiased=False) + torch.pow(a.mean(dim=0) - b.mean(dim=0), 2) + 1e-5)
    return rho

def eval_metrics(y_pred, y_true):
    mae = F.l1_loss(y_pred, y_true, reduction='mean')
    mse = F.mse_loss(y_pred, y_true, reduction='mean')
    rmse = torch.sqrt(mse)
    pcc = PCC(y_pred, y_true)
    ccc = CCC(y_pred, y_true)
    return mae, mse, rmse, pcc, ccc

def logging(mode, output_name, log_writer, loss, mae, mse, rmse, pcc, ccc, idx):
    attrs = {
        'AR': ['Arousal', 'Valence'],
        'ECG': ['ECG_L', 'ECG_R']
    }
    log_writer.add_scalar('{}-Loss/{}'.format(output_name, mode), loss, idx)
    log_writer.add_scalar('{}-MAE/{}'.format(output_name, mode), mae, idx)
    log_writer.add_scalar('{}-MSE/{}'.format(output_name, mode), mse, idx)
    log_writer.add_scalar('{}-RMSE/{}'.format(output_name, mode), rmse, idx)
    if len(ccc) > 1:
        for i in range(len(attrs[output_name])):
            log_writer.add_scalar('{}-PCC/{}'.format(attrs[output_name][i], mode), pcc[i], idx)
            log_writer.add_scalar('{}-CCC/{}'.format(attrs[output_name][i], mode), ccc[i], idx)
    else:
        log_writer.add_scalar('{}-PCC/{}'.format(output_name, mode), pcc, idx)
        log_writer.add_scalar('{}-CCC/{}'.format(output_name, mode), ccc, idx)

def scatter_fn(y, y_pred):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(y, y_pred)
    axs.set_xlabel('True Value', fontsize=10)
    axs.set_ylabel('Predicted Value', fontsize=10)
    return fig

def plot_fn(y, y_pred):
    fig, axs = plt.subplots(1, 1, figsize=(20, 8))
    axs.plot(y, label='Predicted Value')
    axs.plot(y_pred, label='True Value')
    axs.set_xlabel('Overtime Index', fontsize=10)
    axs.set_ylabel('ECG Value', fontsize=10)
    axs.legend(loc='upper right')
    return fig

def eval_dataloader(model_path, val_dataset, batch_size, loader_kwargs, uid=None):
    fname = os.path.basename(model_path).split('.')[0]
    uid = uid or int(fname.replace('pid_', ''))
    val_idx = [idx[0] for idx in val_dataset.idxs if idx[1] == uid]
    val_set = data.Subset(val_dataset, val_idx)
    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=series_collate,
        **loader_kwargs
    )
    print('AMIGO {}: Test samples: {}'.format(uid, len(test_loader)))
    return test_loader

@torch.no_grad()
def run_val(model, test_loader, batch_size):
    y_pred_AR = torch.Tensor().cuda()
    y_true_AR = torch.Tensor().cuda()
    y_pred_ECG = torch.Tensor().cuda()
    y_true_ECG = torch.Tensor().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, labels1, labels2, _) in enumerate(test_loader):
            inputs, labels1, labels2 = inputs.cuda(), labels1.cuda(), labels2.cuda()
            outputs = model(inputs)
            labels = [labels1, labels2]

            y_pred_AR = torch.cat((y_pred_AR, outputs[0]), 0)
            y_true_AR = torch.cat((y_true_AR, labels[0]), 0)
            y_pred_ECG = torch.cat((y_pred_ECG, outputs[1].permute(0, 2, 1).reshape((batch_size * 2560, 2))), 0)
            y_true_ECG = torch.cat((y_true_ECG, labels[1].permute(0, 2, 1).reshape((batch_size * 2560, 2))), 0)
            del inputs, labels1, labels2
    return y_pred_AR, y_true_AR, y_pred_ECG, y_true_ECG

def val_log(log_writer, alpha, scale_factor, y_pred_ARs, y_true_ARs, y_pred_ECGs, y_true_ECGs, idx=0, uid=''):
    losses = []
    mae, mse, rmse, pcc, ccc = eval_metrics(y_pred_ARs, y_true_ARs)
    loss = (1-ccc).mean() + alpha * rmse
    losses.append(loss)
    logging('Validation-{}'.format(uid), 'AR', log_writer, loss, mae, mse, rmse, pcc, ccc, idx)
    mae, mse, rmse, pcc, ccc = eval_metrics(y_pred_ECGs, y_true_ECGs)
    loss = (1-ccc).mean() + alpha * rmse
    losses.append(loss)
    logging('Validation-{}'.format(uid), 'ECG', log_writer, loss, mae, mse, rmse, pcc, ccc, idx)

    y_pred_ARs = y_pred_ARs.permute(1, 0)
    y_true_ARs = y_true_ARs.permute(1, 0)
    y_pred_ECGs = y_pred_ECGs.permute(1, 0)
    y_true_ECGs = y_true_ECGs.permute(1, 0)
    scatter = scatter_fn(y_pred_ARs[0].cpu(), y_true_ARs[0].cpu())
    log_writer.add_figure('Pred vs Actual: {}/{}'.format('Arousal', uid), scatter, idx) 
    scatter = scatter_fn(y_pred_ARs[1].cpu(), y_true_ARs[1].cpu())
    log_writer.add_figure('Pred vs Actual: {}/{}'.format('Valence', uid), scatter, idx)
    scatter = plot_fn(y_pred_ECGs[0].cpu(), y_true_ECGs[0].cpu())
    log_writer.add_figure('Pred vs Actual: {}/{}'.format('ECG_L', uid), scatter, idx)
    scatter = plot_fn(y_pred_ECGs[1].cpu(), y_true_ECGs[1].cpu())
    log_writer.add_figure('Pred vs Actual: {}/{}'.format('ECG_R', uid), scatter, idx)
    return losses