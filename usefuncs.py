import shutil
import torch
import os
from scipy.stats import pearsonr
import numpy as np

def masked_MSEloss(output, target):
    vec = (output - target)**2
    mask = target > -900.0
    loss = torch.sum(vec[mask])/torch.sum(mask)
    return loss

def full_objective(model, inputs, targets, criterion = masked_MSEloss):
    outputs = model(inputs)
    return criterion(outputs, targets)

def compute_predictions(loader, model, reshape=True, stack=True, return_lag=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y, y_hat = [], []
    for x_val, y_val in loader:
        neurons = y_val.size(-1)

        y_mod = model(x_val.to(device)).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat)
    return y, y_hat

def compute_scores(y, y_hat):
    corrs = []
    for i in range(y.shape[1]):
        val_idx = y[:,i] != -999.0
        corrs.append(pearsonr(y[val_idx, i], y_hat[val_idx, i])[0])
    return np.nanmean(corrs)

def save_checkpoint(state, is_best, checkpoint, model_str):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
       state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
       is_best: (bool) True if it is the best model seen till now
       checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last_%s.pth.tar'% model_str)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_%s.pth.tar' % model_str))
