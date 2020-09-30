import torch
from torch.utils.data import Dataset
# utils


class ResmedDatasetEpoch(Dataset):
    """Creates a dataset for resmed streaming data
    """

    def __init__(
        self,
        batch_size: int,
        data: object,
        transform: list = None,
        device: str = 'cuda',
        means: list = None,
        stds: list = None
    ):
        """Epoch Dataset for Resmed Anomaly detector

        Args:
            batch_size (int): batch size
            data (object): location of training data or tensor itself.
            transform (list, optional): transforms; not used yet.
                Defaults to None.
            device (str, optional): device. Defaults to 'cuda'.
            means (list, optional): means used during preprocessing.
                Defaults to None.
            stds (list, optional): stds used during preprocessing.
                Defaults to None.
        """
        super().__init__()
        if isinstance(data, torch.Tensor):
            self.respiration_data = data
        elif isinstance(data, str):
            self.respiration_data = torch.load(data)
        else:
            raise ValueError

        self.batch_size = batch_size
        self.transform = transform
        self.device = device

        # cut length that it matches with batch_size
        self.num_samples = self.respiration_data.shape[0]
        self.num_samples = (self.num_samples // batch_size) * batch_size
        self.respiration_data = self.respiration_data[:self.num_samples, :, :3]

        self.means = self.respiration_data \
            .view(-1, 3) \
            .mean(axis=0) if means is None else means
        self.stds = self.respiration_data \
            .view(-1, 3) \
            .std(axis=0) if stds is None else stds

        self.respiration_data = (
            self.respiration_data - self.means) / (self.stds * 2)

    def __len__(self):
        return(self.num_samples)

    def __getitem__(self, idx):
        if self.device == 'cuda':
            return(self.respiration_data[idx].cuda())
        else:
            return(self.respiration_data[idx])

    def backtransform(self, x):
        return (x * (self.stds * 2)) + self.means

    def get_train_config(self):
        config = {
            "means": self.means.tolist(),
            "stds": self.stds.tolist()
        }
        return config


class TestDataset(Dataset):
    """Creates a dataset for testing
    """

    def __init__(self, batch_count, seq_len, device='cuda'):
        """Initialize Resmed Dataset

        Arguments:
            Dataset {torch.utils.data.Dataset} -- class dataset
            file_name {string} -- where the data should be loaded from
            batch_size {int} -- what batch size to use

        Keyword Arguments:
            transform {string} -- which transformation to use (default: {None})
        """
        super(TestDataset, self).__init__()
        self.batch_count = batch_count
        self.seq_len = seq_len
        self.measure_points = batch_count * seq_len
        self.device = device

        # cut length that it matches with batch_count
        sin_data = torch.sin(torch.arange(0, self.measure_points, 0.1))[
            :self.measure_points]
        self.batched_sin = sin_data.view(self.batch_count, self.seq_len, -1)

    def __len__(self):
        return(self.batch_count)

    def __getitem__(self, idx):
        if self.device == 'cuda':
            return(self.batched_sin[idx].cuda())
        else:
            return(self.batched_sin[idx])
