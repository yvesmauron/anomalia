import os
from abc import ABC, abstractmethod
import mlflow
import mlflow.pytorch
import torch
from azureml.core.run import Run
from azureml.core import Experiment


class MLLogger(ABC):
    """Base class for logging ml experiments
    """
    @abstractmethod
    def __init__(self):
        """constructor
        """
        pass

    @abstractmethod
    def start_run(self):
        """Indicate start of run
        """
        pass

    @abstractmethod
    def log(self):
        """Log a metric of parameter
        """
        pass

    @abstractmethod
    def save_model(self):
        """Save trained model  
        """
        pass

    @abstractmethod
    def end_run(self):
        """Indicate end of run
        """
        pass

class MLFlowLogger(MLLogger):
    """Logger for MLFlow
    """
    def __init__(self, experiment_name, artifact_path, conda_env, code_paths):
        """Constructor
        
        Arguments:
            experimentName {string} -- name of the experiment
            artifact_path {string} -- path to artifact
            conda_env {string} -- path to conda envrionment file
            code_paths {string} -- path to coda directory
        """
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path 
        self.env = conda_env
        self.code_paths = code_paths

        mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        """Indicate start of run
        """
        mlflow.start_run()

    def log(self, key, value, step=None):
        """log a metric of parameter
        
        Arguments:
            key {string} -- name of metric of parameter
            value {string} -- value to be logged
        
        Keyword Arguments:
            step {int} -- e.g. which epoch this loss was recorded (default: {None})
        """ 
        if step == None:
            mlflow.log_param(key, value)
        else:
            mlflow.log_metric(key, value, step)

    def save_model(self, model, flavor='pytorch'):
        """Save trained model  
        
        Arguments:
            model {pytorch.model} -- model to be saved
            flavor {string} -- flavor of model (pytroch, tensorflow)
        """
        if flavor == 'pytorch':
            mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=self.artifact_path, 
                    conda_env=self.env, 
                    code_paths=self.code_paths
                )
        else:
            NotImplementedError

    def end_run(self):
        """Indicate end of run
        """
        mlflow.end_run()


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
        self.workspace= workspace
            
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

