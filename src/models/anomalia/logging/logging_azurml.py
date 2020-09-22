import os

from azureml.core.run import Run
from azureml.core import Experiment
import torch

from .logging import MLLogger

class AzureLogger(MLLogger):
    """Logger class for azure
    """

    def __init__(self, workspace, experiment_name, artifact_path):
        """Constructor

        Arguments:
            run {Run} -- object Run
            artifact_path {string} -- path to artifact
        """
        self.artifact_path = artifact_path
        self.experiment_name = experiment_name
        self.workspace = workspace

        self.experiment = Experiment(workspace, name=experiment_name)

    def start_run(self):
        """Indicate start of run
        """
        self.run = Run.get_context()

    def log(self, key, value, step=None):
        """Log metric or parameter

        Arguments:
            key {string} -- name or metric to be logged
            value {int/string} -- value to be logged    

        Keyword Arguments:
            step {int} -- not used (default: {None})
        """
        self.run.log(key, value)

    def save_model(self, model, flavor='pytorch'):
        """Save trained model  

        Arguments:
            model {pytorch.model} -- model to be saved
            flavor {string} -- flavor of model (pytroch, tensorflow)
        """
        if flavor == 'pytorch':
            # note that repeated saving could be implemented
            torch.save(model, os.path.join(self.artifact_path, 'model.pt'))
        else:
            NotImplementedError

    def end_run(self):
        """Indicate end of run
        """
        # not necessary
        pass
