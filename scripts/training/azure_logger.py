from azureml.core.run import Run

class AzureLogger:
    def __init__(self, experimentName):
        self.Name = experimentName
        self.Run = Run.get_context()


    def log(self, key, value, step=None):
        if step == None:
            self.Run.log_param(key, value)


    def saveModel(self, model):
        a = True
        # Todo