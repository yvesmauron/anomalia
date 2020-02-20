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

import numpy as np

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()

test_mode = False


if __name__ == "__main__":

    ####################################################
    # parametrization
    # data loader
    batch_size = 64
    train_path = '/train_resmed.pt'
    # smavra
    smarva_input_params = {
        'input_size':1 if test_mode else 3,
        'hidden_size':10 if test_mode else 30,
        'latent_size':1 if test_mode else 3,
        'attention_size':1 if test_mode else 3,
        'output_size':1 if test_mode else 3,
        'num_layers':1 if test_mode else 2,
        'n_heads':1 if test_mode else 3,
        'dropout':0.25,
        'batch_first':True,
        'cuda':True,
        'mode':'static',
        'rnn_type':'LSTM'
    }

    # trainer
    lr=0.0005
    clip = True # options: True, False
    max_grad_norm=5
    kld_annealing_start_epoch = 0
    kld_annealing_max = 0.7
    kld_annealing_intervals=[15, 10, 5]
    kld_latent_loss_weight=1
    kld_attention_loss_weight=.5

    # training duration
    n_epochs = 400
    #
    if test_mode:
        dataset = TestDataset(200, 200)
    else:
        dataset = ResmedDatasetEpoch(train_path, batch_size)

    num_batches = dataset.__len__() // batch_size

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    smavra = SMAVRA(
        **smarva_input_params
    )

    #
    ## Define Loss, Optimizer
    optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)

    smavra.train()

    smavra.cuda()

    mask = None #torch.ones(X_padded.shape).masked_fill(X_padded == 0, 0)

    # todo:
    # - kld annealing
    # - weightening kld latent versus kld attention
    # - put wights into metric or param

    def kld_annealing(start_epoch, n_epochs, epoch, intervals=[3, 4, 3], max_weight=1):
        if len(intervals) != 3:
            ValueError

        intervals = np.array(intervals)
        iteration_cycle = intervals.sum()
        phase_bounds = intervals.cumsum()
        growth_speed = max_weight / intervals[1]
        
        current_state = epoch % iteration_cycle
        if current_state < intervals[0] or epoch < start_epoch:
            return 0
        elif intervals[0] <= current_state and current_state < phase_bounds[1]:
            return (current_state - intervals[0]) * growth_speed
        else:
            return max_weight


    for key, value in smarva_input_params.items():
        run.log(key, value)
        run.log('n_epochs', n_epochs)
        run.log('kld_annealing_start_epoch', kld_annealing_start_epoch)
        run.log('kld_annealing_max', kld_annealing_max)
        run.log('kld_annealing_intervals', kld_annealing_intervals)
        run.log('kld_latent_loss_weight',  kld_latent_loss_weight)
        run.log('kld_attention_loss_weight',  kld_attention_loss_weight)

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
            if test_mode:
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
        
        run.log('KLD-Annealing weight', kld_error_weight, step=epoch)
        run.log('Reconstruction Loss', (epoch_recon_loss / t), step=epoch)
        run.log('KLD-Latent Loss', (epoch_kld_latent_loss / t), step=epoch)
        run.log('KLD-Attention Loss', (epoch_kld_attention_loss / t), step=epoch)
        run.log('Loss', (epoch_loss / t), step=epoch)

    torch.save(smavra, os.path.join('model', 'smavra.pt'))
        #       #if (t + 1) % print_every == 0:
        #       print('Batch: {}/{}............. \n'.format(i, num_batches), end=' ')
        #       print(
        #           'Recon: {:.4f} \n KLD-Latent: {:.4f} \n KLD-Attention: {:.4f} \n Loss: {:.4f} \n'.format(
        #               recon_loss,
        #               kld_latent_loss,
        #               kld_attention_loss,
        #               loss
        #           )
        #       )
    #
        #   print('Average loss: {:.4f}'.format(epoch_loss / t))


    #smavra = SMAVRA(**smarva_input_params)
    #smavra.load_state_dict(torch.load(os.path.join(model_path, 'smavrae.pt')))
    #    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #    model_dir_path = os.path.join('models', 'smavrae', date_time)
    #    
    #    if not os.path.exists(model_dir_path):
    #        os.makedirs(model_dir_path)
    #    
    #    model_path = os.path.join(model_dir_path, 'smavrae.pt')
    #
    #    torch.save(smavra, model_path)

    #mlflow.pytorch.load_model('runs:/14fdf9cade444e24834a41e87bc14bfd/model')