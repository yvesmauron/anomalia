
from abc import ABC, abstractmethod


class MLTracker(ABC):
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
