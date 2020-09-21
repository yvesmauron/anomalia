# general imports
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from glob import glob

from anomalia.resmed_preprocess import *
from anomalia.utils import *

from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from anomalia.resmed.preprocess import *
from torch.utils.data import DataLoader
import shutil

import torch
import json

from plotnine import *

import mlflow
import mlflow.pytorch

from mlflow.tracking import MlflowClient

test_mode = False

run_id = 'cc8673900bf24e49842f527591ffe813'
score_dir = "data/resmed/staging/20200427/"
pred_path = "data/preds/cleansing"
seq_len = 750
batch_size = 124

# ----------------------------------------
# connect to mlflow
mlflow_client = MlflowClient()

# ----------------------------------------
# download artifact
temp_dir = "./tmp"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

os.makedirs(temp_dir) 
preprocessing_config = mlflow_client.download_artifacts(run_id, "config/data_config.json", temp_dir)
resmed_config = mlflow_client.download_artifacts(run_id, "config/resmed.json", temp_dir)

# read artifacts
with open(preprocessing_config, "r") as f:
    pp_config = json.load(f)

# read artifacts
with open(resmed_config, "r") as f:
    data_config = json.load(f)

respiration_data, train_df = read_data(score_dir, data_config=resmed_config, scoring=True)


# count = 0
# for i in range(len(respiration_data)):
#     count += respiration_data[i].shape[0]

# train_df.shape[0]
# count * 750

predict_dataset = ResmedDatasetEpoch(
    batch_size=1,
    respiration_data=respiration_data,
    means=torch.tensor(pp_config["means"]),
    stds=torch.tensor(pp_config["means"])
)

predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

smavra = mlflow.pytorch.load_model('runs:/' + run_id + '/model')

smavra.eval()
smavra.cuda()

preds = []

num_cols = [_["name"] for _ in data_config["numerical_columns"]]

with tqdm(total=len(predict_loader), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
    for i, epoch in enumerate(predict_loader):
        pbar.set_postfix(file=i)
        # get prediction
        mu_scaled, _ = smavra(epoch)
        # reshape params
        batch, seq, fe = mu_scaled.shape
        # move it back to cpu
        # mse of epoch
        epoch_mse = np.repeat(
            torch.pow((mu_scaled - epoch), 2).mean(axis=(1,2)).cpu().detach().numpy(), 
            seq_len
        )
        epoch_mse = epoch_mse.reshape(batch * seq, 1)
        # mse of timestep
        t_mse = torch.pow(
            (mu_scaled - epoch), 2
        ).mean(axis=(2)).cpu().detach().numpy()
        t_mse = t_mse.reshape(batch * seq, 1)
        # se of timestep and measure
        m_se = torch.pow((mu_scaled - epoch), 2)
        m_se = m_se.view(batch * seq, fe).cpu().detach().numpy()

        mu = predict_dataset.backtransform(mu_scaled.view(batch * seq, fe).cpu().detach()).squeeze(0).numpy()
        mu_scaled = mu_scaled.view(batch * seq, fe).cpu().detach().numpy()
        
        predictions = np.concatenate([mu_scaled, mu, epoch_mse, t_mse,  m_se], axis=1)
        
        colnames = ["epoch_mse", "t_mse"] \
            + [f"{_}_se" for _ in num_cols] \
            + [f"{_}_mu_scaled" for _ in num_cols] \
            + [f"{_}_mu" for _ in num_cols] 
        predictions = pd.DataFrame(predictions, columns=colnames)

        preds.append(predictions)

        pbar.update(1)

if len(predict_dataset) > 1:
    preds = pd.concat(preds, ignore_index=True)
else:
    preds = preds[0]


preds = pd.concat([train_df, preds], axis=1)

if os.path.exists(pred_path):
    shutil.rmtree(pred_path)

os.makedirs(pred_path)

preds.to_csv(os.path.join(pred_path, "preds.csv"), index=False)
# batch_threshold = 0.5
# point_threshold = 0.12

# train_df["batch_recon_error"] = np.repeat(batch_recon_error, seq_len)
# train_df["batch_is_idle"] = np.repeat((batch_recon_error < batch_threshold).astype('float'), seq_len)

# train_df["point_recon_error"] = point_recon_error
# train_df["point_is_idle"] = (point_recon_error < point_threshold).astype('float')



# train_df_limited = train_df.iloc[:(750*5),:]
# (
#     ggplot(train_df, aes(x = 'TimeStamp', y = numerical_columns[0], color="batch_recon_error")) +
#     geom_line()
# )

# clean up dir
shutil.rmtree(temp_dir)