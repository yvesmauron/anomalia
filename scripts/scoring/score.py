import os
import sys
sys.path.append(os.getcwd())

from atemteurer.Layers import SMAVRA
from atemteurer.Datasets import ResmedDatasetEpoch, TestDataset
from torch.utils.data import DataLoader

import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

import os
import numpy as np

import mlflow
import mlflow.pytorch

test_mode = False

run_id = '859803c3d01e4eb4bfae47806cca67c8'

if test_mode:
    mlflow.set_experiment('SMARVA_Test')
else:
    mlflow.set_experiment('SMARVA')

####################################################
# parametrization
# data loader
train_path = 'data/resmed/train/train_resmed.pt'
batch_size = 1
device = 'cuda'
dataset = ResmedDatasetEpoch(train_path, batch_size)

train_means = dataset.means
train_std = dataset.stds

test_dataset = ResmedDatasetEpoch('data/resmed/test/test_resmed.pt', batch_size, means=train_means, stds=train_std)
entire_test_set = test_dataset.respiration_data
eval_dataset = ResmedDatasetEpoch('data/resmed/eval/eval_resmed.pt', batch_size, means=train_means, stds=train_std)
entire_eval_set = eval_dataset.respiration_data[:20,:,:]

if device == 'cuda':
    entire_test_set = entire_test_set.cuda()
    entire_eval_set = entire_eval_set.cuda()

smavra = mlflow.pytorch.load_model('runs:/' + run_id + '/model')

smavra.eval()

evaluate = False

if evaluate:
    data = entire_eval_set
else:
    data = entire_test_set

preds = smavra(data)

# line plot
idx = 2
col = 0
plot_data = torch.stack((data[idx,:,col], preds[idx,:,col]), dim=1).cpu().detach().numpy()

plt.plot(plot_data)
plt.show()

# compare loss with training loss
recon_loss, kld_latent_loss, kld_attention_loss = smavra.compute_loss(data, preds)
print('loss:', recon_loss.item())

# illustrate attention weights
h_t, latent, attention_weights, attention, lengths = smavra.encode(data[:5,:,:])

attention_head = attention_weights[idx,1,:,:].squeeze().cpu().detach().numpy()
plt.matshow(attention_head)
plt.show()

plt.hist(latent.squeeze().cpu().detach().numpy())
plt.show()



torch.load('./data/test/test_re')