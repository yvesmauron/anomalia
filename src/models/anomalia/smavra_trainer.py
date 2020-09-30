import os
from src.models.anomalia.trainer import Trainer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime as dt


class SmavraTrainer(Trainer):
    """Trainer for Smavra"""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        optimizer: object,
        logger: object,
        checkpoint_interval: int = 1,
        checkpoint_path: str = None
    ):
        """Constructor

        Args:
            model (torch.nn.Module): model to be trained
            dataset (Dataset): dataset for training
            optimizer (object): optimizer for training
            logger (object): logger (mlflow or Azure)
            checkpoint_interval (int, optional): frequency of checkpoint.
                Defaults to 1.
            checkpoint_path (str, optional): where to checkpoint.
                Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval

        if checkpoint_path is None:
            self.checkpoint_path = os.path.join(
                'models', dt.now().strftime("%Y%m%d_%H%M%S"))
        else:
            self.checkpoint_path = checkpoint_path

            # check if arguments are valid, i.e. if directory exists
        if os.path.exists(self.checkpoint_path):
            FileExistsError
        else:
            os.makedirs(self.checkpoint_path)

    def fit(
        self,
        n_epochs: int,
        batch_size: int,
        clip: bool = True,
        max_grad_norm: int = 5,
        kld_annealing_start_epoch: int = 0,
        kld_annealing_max: float = 0.8,
        kld_annealing_intervals: list = [15, 30, 5],
        kld_latent_loss_weight: float = .6,
        kld_attention_loss_weight: float = .3
    ) -> torch.nn.Module:
        """fit model

        Args:
            n_epochs (int): number of epochs
            batch_size (int): batch size
            clip (bool, optional): gradient clip. Defaults to True.
            max_grad_norm (int, optional): max grad norm. Defaults to 5.
            kld_annealing_start_epoch (int, optional): kld start after epoch.
                Defaults to 0.
            kld_annealing_max (float, optional): kld max. Defaults to 0.8.
            kld_annealing_intervals (list, optional): kld interval.
                Defaults to [15, 30, 5].
            kld_latent_loss_weight (float, optional): kld latent loss weight.
                Defaults to .6.
            kld_attention_loss_weight (float, optional): kld attention loss.
                Defaults to .3.

        Returns:
            torch.nn.Module: trained model
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

                batch_true = batch[:, :, :3]

                recon_loss, kld_latent_loss, kld_attention_loss = \
                    self.model.compute_loss(out, batch_true, mask=None)

                loss = (
                    recon_loss +
                    (
                        kld_error_weight * (
                            kld_latent_loss_weight * kld_latent_loss
                            +
                            kld_attention_loss_weight * kld_attention_loss
                        )
                    )
                )

                loss.backward()  # Does backpropagation

                if clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_grad_norm)

                # accumulator
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.data.item()
                epoch_kld_latent_loss += kld_latent_loss.data.item()
                epoch_kld_attention_loss += kld_attention_loss \
                    if isinstance(kld_attention_loss, int) \
                    else kld_attention_loss.data.item()

                self.optimizer.step()

                t += 1

            self.logger.log('KLD-Annealing weight',
                            kld_error_weight, step=epoch)
            self.logger.log('Reconstruction Loss',
                            (epoch_recon_loss / t), step=epoch)
            self.logger.log('KLD-Latent Loss',
                            (epoch_kld_latent_loss / t), step=epoch)
            self.logger.log('KLD-Attention Loss',
                            (epoch_kld_attention_loss / t), step=epoch)
            self.logger.log('Loss', (epoch_loss / t), step=epoch)

            # set checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.checkpoint(epoch, epoch_loss)

        self.logger.save_model(self.model)

        return self.model

    def kld_annealing(
        self,
        start_epoch: int,
        n_epochs: int,
        epoch: int,
        intervals: list = [3, 4, 3],
        max_weight: int = 1
    ) -> float:
        """kld annealing function

        see: https://www.microsoft.com/en-us/research/blog/less-pain-more-
             gain-a-simple-method-for-vae-training-with-less-of-
             that-kl-vanishing-agony/
        Args:
            start_epoch (int): which epoch annealing should start
            n_epochs (int):  number of epochs in the training
            epoch (int):  current epoch number
            intervals (list):  list with intervals for annealing
                [beta= 0, beta growing, beta = max_weight].
                Defaults to [3, 4, 3].
            max_weight (float):  maximum possible weight.
                Defaults to 1

        Returns:
            float: kld weight
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

    def checkpoint(self, epoch: int, loss: float):
        """Save checkpoint

        Args:
            epoch (int): epoch of this checkpointS
            loss (float): loss of this epoch
        """

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, os.path.join(
            self.checkpoint_path,
            "epoch_{:04d}_loss_{:.5f}".format(epoch, loss) + ".pt"
        )
        )
