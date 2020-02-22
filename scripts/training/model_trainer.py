import os
import sys
sys.path.append(os.getcwd())

from anomalia.layers import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from anomalia.logging import AzureLogger, MLFlowLogger
from torch.utils.data import DataLoader

import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils
import torch.nn.functional as F
from datetime import datetime

import numpy as np

class ModelTrainer():
    def __init__(self, logger):
        self.logger = logger
        self.test_mode = False
        self.device = 'cuda'

    def run(self, train_path):


        ####################################################
        # parametrization
        # data loader

        # smavra


        # training duration

        num_batches = dataset.__len__() // batch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        #
        ## Define Loss, Optimizer
        optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)

        smavra.train()

        if self.device == 'cuda':
            smavra.cuda()


        # todo:
        # - kld annealing
        # - weightening kld latent versus kld attention
        # - put wights into metric or param




        

            for key, value in smarva_input_params.items():
                logger.log_param(key, value)

            logger.log_param('n_epochs', n_epochs)
            logger.log_param('kld_annealing_start_epoch', kld_annealing_start_epoch)
            logger.log_param('kld_annealing_max', kld_annealing_max)
            logger.log_param('kld_annealing_intervals', kld_annealing_intervals)
            logger.log_param('kld_latent_loss_weight',  kld_latent_loss_weight)
            logger.log_param('kld_attention_loss_weight',  kld_attention_loss_weight)

            for epoch in range(n_epochs):

                epoch_loss = 0
                epoch_recon_loss = 0
                epoch_kld_latent_loss = 0
                epoch_kld_attention_loss = 0
                t = 0

                kld_error_weight = kld_annealing(kld_annealing_start_epoch, n_epochs, epoch, intervals=kld_annealing_intervals, max_weight=kld_annealing_max)

                for i, batch in enumerate(loader):

                    optimizer.zero_grad()
                    out = smavra(batch)

                    if self.TestMode:
                        batch_true = batch
                    else:
                        batch_true = batch[:,:,:3]
                    
                    recon_loss, kld_latent_loss, kld_attention_loss = smavra.compute_loss(out, batch_true, mask=mask)


                    loss = (recon_loss) + (kld_error_weight*(kld_latent_loss_weight*kld_latent_loss + kld_attention_loss_weight*kld_attention_loss))

                    loss.backward() # Does backpropagation and calculates gradients

                    if clip:
                        torch.nn.utils.clip_grad_norm_(smavra.parameters(), max_norm = max_grad_norm)

                    # accumulator
                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.data.item()
                    epoch_kld_latent_loss += kld_latent_loss.data.item()
                    epoch_kld_attention_loss += kld_attention_loss if isinstance(kld_attention_loss, int) else kld_attention_loss.data.item()
                    
                    optimizer.step()

                    t += 1
                
                logger.log_metric('KLD-Annealing weight', kld_error_weight, step=epoch)
                logger.log_metric('Reconstruction Loss', (epoch_recon_loss / t), step=epoch)
                logger.log_metric('KLD-Latent Loss', (epoch_kld_latent_loss / t), step=epoch)
                logger.log_metric('KLD-Attention Loss', (epoch_kld_attention_loss / t), step=epoch)
                logger.log_metric('Loss', (epoch_loss / t), step=epoch)
            
            logger.save_model(smavra)

        return smavra
