import mlflow
import mlflow.pytorch

class MLFlowLogger:
    def __init__(self, experimentName, artifact_path, conda_env, code_paths):
        self.name = experimentName
        self.artifact_path = artifact_path 
        self.env = conda_env
        self.code_paths = code_paths

        mlflow.set_experiment(self.name)

    def log(self, key, value, step=None):
        if step == None:
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, value, step)

    def saveModel(self, model):
        mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=self.artifact_path, 
                conda_env=self.env, 
                code_paths=self.code_paths
            )