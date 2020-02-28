import os
import sys
sys.path.append(os.getcwd())

from anomalia.layers import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from torch.utils.data import DataLoader

import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

import shutil
import glob
import numpy as np
import pandas as pd
from plotnine import *

import mlflow
import mlflow.pytorch

test_mode = False

run_id = '695a350b357f42debd61e0a744a28335'
mlflow.set_experiment('SMARVA')

# ------------------------------------
# get checkpoints from directory
training_run = 'models/20200227_195257/'
ordered_checkpoint_list = sorted(glob.glob(training_run + '*.pt'), key=os.path.getmtime)

with open('order.txt', 'w') as filehandle:
    for listitem in ordered_checkpoint_list:
        filehandle.write('%s\n' % listitem)

# define directory for writing the plots
model_fitting_data_dir = 'data/resmed/analysis/model_fitting'

# check if arguments are valid, i.e. if directory exists
if os.path.exists(model_fitting_data_dir):
    shutil.rmtree(model_fitting_data_dir)

# create directoy ; make sure everything is empty
os.makedirs(model_fitting_data_dir)

train_path = 'data/resmed/train/train_resmed.pt'
batch_size = 1
device = 'cuda'
dataset = ResmedDatasetEpoch(train_path, batch_size)

smavra = mlflow.pytorch.load_model('runs:/' + run_id + '/model')

smavra.eval()

# show only one example
sample_idx = 2
sample = dataset.respiration_data[sample_idx,:,:].unsqueeze(0)

def tensor_to_data_frame(tensor, epoch, device, result_type):
    epoch = torch.FloatTensor([[epoch]]).to(device)
    tensor = torch.cat((tensor.squeeze(0), epoch.repeat(750,1)), 1)
    tensor = pd.DataFrame(tensor.detach().cpu().numpy())
    tensor.columns = ['MaskPressure', 'RespFlow', 'DeliveredVolume', 'Epoch']
    tensor['Type'] = result_type
    tensor['Index'] = tensor.index
    return(tensor)

for epoch in range(len(ordered_checkpoint_list)):

    checkpoint = torch.load(ordered_checkpoint_list[epoch])
    smavra.load_state_dict(checkpoint['model_state_dict'])
    smavra.to(device)
    sample = sample.to(device)

    with torch.no_grad():
        preds = smavra(sample)
        recon_loss, _, _ = smavra.compute_loss(sample, preds)
        rec_loss = recon_loss.item()
        preds = tensor_to_data_frame(preds, epoch, device, "Predictions")
        true = tensor_to_data_frame(sample, epoch, device, "True")
        plot_data = true.append(preds)
        plot_data =pd.melt(plot_data, id_vars=['Index', 'Type','Epoch'], value_vars=['MaskPressure', 'RespFlow', 'DeliveredVolume'])
        plot = (
            ggplot(plot_data) +
            aes(x = 'Index', y = 'value', color='factor(Type)') +
            geom_line() +
            facet_wrap('~variable', ncol=1, nrow=3) +
            labs(color='Type', x='Epoch: {:.0f}; Loss: {:.4f}'.format(epoch, rec_loss)) +
            theme(
                axis_title_y = element_blank()
            )
        )
        plot.save(filename = os.path.join(model_fitting_data_dir, '{:04d}.png'.format(epoch)), height=15, width=20, units = 'cm', dpi=250) 





train_means = dataset.means
train_std = dataset.stds

test_dataset = ResmedDatasetEpoch('data/resmed/test/test_resmed.pt', batch_size, means=train_means, stds=train_std)
entire_test_set = test_dataset.respiration_data[:20,:,:]
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


