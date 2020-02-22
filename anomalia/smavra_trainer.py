from anomalia.trainer import Trainer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import inspect


class SmavraTrainer(Trainer):
    """Trainer for Smavra
    
    Arguments:
        Trainer {anomalia.trainer.Trainer} -- base class
    """
    def __init__(self, model, dataset, optimizer, logger):
        """Constructor
        
        Arguments:
            model {anomalia.smavra.SMAVRA} -- SMAVRA Model
            train_data {string} -- path to train data directory
            logger {anomalia.MLLogger} -- subclass of MLLogger
        """
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.logger = logger
    
    def fit(
        self, 
        n_epochs,
        batch_size,
        clip = True, 
        max_grad_norm=5,
        kld_annealing_start_epoch = 0,
        kld_annealing_max = 0.7,
        kld_annealing_intervals = [15, 10, 5],
        kld_latent_loss_weight=1,
        kld_attention_loss_weight=.5
        ):
        """[summary]
        
        Arguments:
            n_epochs {[type]} -- [description]
            batch_size {[type]} -- [description]
        
        Keyword Arguments:
            lr {float} -- [description] (default: {0.0005})
            clip {bool} -- [description] (default: {True})
            max_grad_norm {int} -- [description] (default: {5})
            kld_annealing_config {[type]} -- [description] (default: {None})
            kld_latent_loss_weight {int} -- [description] (default: {1})
            kld_attention_loss_weight {float} -- [description] (default: {.5})
        """
        # log input parameters
        param_dict = dict(locals())
        param_dict['self'] = 'SMAVRA'

        for key, value in param_dict.items():
            if key != 'self':
                self.logger.log(key, str(value))

        # get number of batches
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # set model into train mode
        self.model.train()

        # iterate over epochs
        for epoch in range(n_epochs):

            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kld_latent_loss = 0
            epoch_kld_attention_loss = 0
            t = 0

            kld_error_weight = self.kld_annealing(
                kld_annealing_start_epoch, 
                n_epochs, 
                epoch, 
                intervals=kld_annealing_intervals, 
                max_weight=kld_annealing_max
            )

            for _, batch in enumerate(loader):

                self.optimizer.zero_grad()
                out = self.model(batch)

                batch_true = batch[:,:,:3]
                
                recon_loss, kld_latent_loss, kld_attention_loss = self.model.compute_loss(out, batch_true, mask=None)

                loss = (
                    recon_loss + 
                    (
                        kld_error_weight * (
                            kld_latent_loss_weight * kld_latent_loss
                            + 
                            kld_attention_loss_weight* kld_attention_loss
                        )
                    )
                
                )

                loss.backward() # Does backpropagation and calculates gradients

                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)

                # accumulator
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.data.item()
                epoch_kld_latent_loss += kld_latent_loss.data.item()
                epoch_kld_attention_loss += kld_attention_loss if isinstance(kld_attention_loss, int) else kld_attention_loss.data.item()
                
                self.optimizer.step()

                t += 1
            
            self.logger.log('KLD-Annealing weight', kld_error_weight, step=epoch)
            self.logger.log('Reconstruction Loss', (epoch_recon_loss / t), step=epoch)
            self.logger.log('KLD-Latent Loss', (epoch_kld_latent_loss / t), step=epoch)
            self.logger.log('KLD-Attention Loss', (epoch_kld_attention_loss / t), step=epoch)
            self.logger.log('Loss', (epoch_loss / t), step=epoch)
        
        self.logger.save_model(self.model)

        return self.model

    def kld_annealing(self, start_epoch, n_epochs, epoch, intervals=[3, 4, 3], max_weight=1):
        """kld annealing function
        
        see: https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
        Arguments:
            start_epoch {[type]} -- which epoch annealing should start
            n_epochs {[type]} -- number of epochs in the training
            epoch {[type]} -- current epoch number
        
        Keyword Arguments:
            intervals {list} -- list with intervals for annealing [beta=0, beta growing, beta=max_weight] (default: {[3, 4, 3]})
            max_weight {int} -- maximum possible weight (default: {1})
        
        Returns:
            [float] -- beta (kld weight)
        """
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
    
    def validate(self):
        pass