from abc import ABC, abstractmethod


class Trainer(ABC):
    """Trainer class to be subclassed
    """
    @abstractmethod
    def __init__(self):
        """Constructor
        """
        pass

    @abstractmethod
    def fit(self):
        """Model fitting
        """
        pass

    @abstractmethod
    def validate(self):
        """Model validation
        """
        pass
