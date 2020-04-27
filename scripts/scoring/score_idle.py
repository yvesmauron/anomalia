# general imports
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from anomalia.resmed_preprocess import *
from anomalia.utils import *

from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from torch.utils.data import DataLoader

import torch
import json

from plotnine import *

f = pd.read_csv("data/resmed/staging/20200427/20200124_120001_0_HRD.edf.csv")

train_df, numerical_columns, one_hot_columns = preprocess_resmed_from_config(f, "config/resmed.json", False)

start, end = get_id_bounds(torch.tensor(train_df[numerical_columns[0]]), -3276.8)

train_df = train_df.iloc[int(start):int(end), :]

seq_len = 750
batch_size = (train_df.shape[0] // seq_len)
end_test = batch_size * seq_len

train_df = train_df.iloc[:end_test, :]

train_input = torch.FloatTensor(train_df[numerical_columns].to_numpy()).view(batch_size, seq_len, 3)

# eval

import mlflow
import mlflow.pytorch

test_mode = False

run_id = 'f2ae0b6d6725487c84e83be458e84424'

if test_mode:
    mlflow.set_experiment('SMARVA_Test')
else:
    mlflow.set_experiment('SMARVA_IDLE')


train_path = 'data/resmed/train/train_resmed.pt'
batch_size = 1
device = 'cuda'
dataset = ResmedDatasetEpoch(train_path, batch_size)
#dataset = dataset.respiration_data[20,:,:]

train_means = dataset.means
train_std = dataset.stds

train_input = (train_input - train_means) / (train_std)


smavra = mlflow.pytorch.load_model('runs:/' + run_id + '/model')

smavra.eval()

smavra.cuda()
train_input = train_input

mus = None

for i in range(train_input.shape[0]):
    print(f"processing batch {i}")
    x_test = train_input[i].unsqueeze(0).cuda()
    mu, _ = smavra(x_test)
    mu = mu.cpu().detach()
    if mus is not None:
        mus = torch.cat([mus, mu], dim=0)
    else:
        mus = mu

batch_recon_error = torch.pow((mus - train_input), 2).mean(axis=(1,2)).cpu().detach().numpy()
point_recon_error = torch.pow((mus - train_input), 2).mean(axis=(2)).cpu().detach().numpy().flatten()

batch_threshold = 0.5
point_threshold = 0.12

train_df["batch_recon_error"] = np.repeat(batch_recon_error, seq_len)
train_df["batch_is_idle"] = np.repeat((batch_recon_error < batch_threshold).astype('float'), seq_len)

train_df["point_recon_error"] = point_recon_error
train_df["point_is_idle"] = (point_recon_error < point_threshold).astype('float')



train_df_limited = train_df.iloc[:(750*5),:]
(
    ggplot(train_df, aes(x = 'TimeStamp', y = numerical_columns[0], color="batch_recon_error")) +
    geom_line()
)