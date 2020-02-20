import torch
from torch.utils.data import Dataset
# scale transformations
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
# utils
import os
import logging
import logging.config
# custom libraries
import atemteurer.utils as utils

# onehot encoding
# batch_size = 3
# seq_len = 3
# n_classes = 3
# # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
# y = torch.LongTensor([[0,0,0],[1,1,1],[2,2,2]]).unsqueeze(2)
# # One hot encoding buffer that you create out of the loop and just keep reusing
# y_onehot = torch.FloatTensor(batch_size, seq_len, n_classes)

# # In your for loop
# y_onehot.zero_()
# y_onehot.scatter_(2, y, 1)

# print(y)
# print(y_onehot)

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('atemteurer')


class ResmedDatasetEpoch(Dataset):
    """Creates a dataset for resmed streaming data
    """
    def __init__(self, file_name, batch_size, transform=None, device='cuda', means=None, stds=None):
        """Initialize Resmed Dataset
        
        Arguments:
            Dataset {torch.utils.data.Dataset} -- class dataset
            file_name {string} -- where the data should be loaded from
            batch_size {int} -- what batch size to use
        
        Keyword Arguments:
            transform {string} -- which transformation to use (default: {None})
        """
        super().__init__()
        self.respiration_data = torch.load(file_name)
        self.batch_size = batch_size
        self.transform = transform
        self.device = device

        # cut length that it matches with batch_size
        self.respiration_data = torch.stack(self.respiration_data, dim=0)
        self.num_samples = self.respiration_data.shape[0]
        self.num_samples = (self.num_samples // batch_size) * batch_size
        self.respiration_data = self.respiration_data[:self.num_samples, :, :3]

        self.means = self.respiration_data.view(-1,3).mean(axis=0) if means is None else means
        self.stds = self.respiration_data.view(-1,3).std(axis=0) if stds is None else stds

        self.respiration_data = (self.respiration_data - self.means) / (self.stds * 2)

    def __len__(self):
        return(self.num_samples)

    def __getitem__(self, idx):
        if self.device == 'cuda':
            return(self.respiration_data[idx].cuda())
        else:
            return(self.respiration_data[idx])
    
    def backtransform(self, x):
        return (x * (self.stds * 2)) + self.means



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
        sin_data = torch.sin(torch.arange(0, self.measure_points, 0.1))[:self.measure_points]
        self.batched_sin = sin_data.view(self.batch_count, self.seq_len, -1)

    def __len__(self):
        return(self.batch_count)

    def __getitem__(self, idx):
        if self.device == 'cuda':
            return(self.batched_sin[idx].cuda())
        else:
            return(self.batched_sin[idx])




#test_dataset = ResmedDatasetEpoch('data/resmed/train/train_resmed.pt', 64)
#
#for i, tensor in enumerate(test_dataset):
#    print('id: {}; tensor.shape: {}'.format(i, tensor.shape))
#
#
#
#test = [1,2,3,4,5]
#len(test) // 2
#test[:2*2]
#
#test = ResmedDatasetSimple('data/resmed/train/train_resmed.pt', 64)
#
#for i in enumerate(test):
#    print(i)
#    input("Press Enter to continue...")
#
#breath_data = torch.load('data/resmed/train/train_resmed.pt')
#
#padded = torch.nn.utils.rnn.pad_sequence(breath_data, batch_first=True)
#lengths = [_.size(0) for _ in breath_data]
#padded = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
#
#breath_data[1].size(0)
#test = torch.load('data/resmed/train/train_resmed.pt')[0:10]
#
#
#test = test[:10]
#
#
#test =  [
#    torch.tensor([[
#        [
#            1,2,3
#        ],
#        [
#            2,3,4
#        ]
#    ],
#    [
#        [
#            1,2,3
#        ],
#        [
#            2,3,4
#        ]
#    ]]),
#    torch.tensor([[
#        [
#            1,2,3
#        ],
#        [
#            2,3,4
#        ]
#    ]]),
#]
#
#torch.cat(test, dim=0)
#
#
#test_padded = torch.nn.utils.rnn.pad_sequence(test, batch_first=True, padding_value=0)
#lengths = [_.size(0) for _ in test]
#torch.nn.utils.rnn.pack_padded_sequence(test_padded, lengths, batch_first=True, enforce_sorted=False)
#
#
#lens = [s.shape[0] for s in test]
#
#plt.hist(lengths)
#plt.show()
#
#plt.boxplot(lens)
#plt.show()
#
#
#from torch import nn
#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
#x_seq = [torch.tensor([5, 18, 29]), torch.tensor([32, 100]), torch.tensor([699, 6, 9, 17])]
#x_padded = pad_sequence(x_seq, batch_first=True, padding_value=0)
#
#pack_padded_sequence(x_padded)
#
#
#
#
#class ResmedDataset(Dataset):
#    """Creates a dataset for resmed streaming data
#    """
#    def __init__(self, file_name, batch_size, max_length=1000000, transform=None, device='cuda'):
#        """Initialize Resmed Dataset
#        
#        Arguments:
#            Dataset {torch.utils.data.Dataset} -- class dataset
#            file_name {string} -- where the data should be loaded from
#            batch_size {int} -- what batch size to use
#        
#        Keyword Arguments:
#            transform {string} -- which transformation to use (default: {None})
#        """
#        super().__init__()
#        self.respiration_data = torch.load(file_name)
#        self.batch_size = batch_size
#        self.transform = transform
#        self.device = device
#
#        # padd breath to have equal length
#        self.padded = torch.nn.utils.rnn.pad_sequence(breath_data, batch_first=True)
#        self.lengths = [_.size(0) for _ in breath_data]
#
#    def __len__(self):
#        return(len(self.lengths))
#
#    def __getitem__(self, idx):
#        padded_x = self.padded[idx]
#        packed_padded = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
#        if self.device == 'cuda':
#            return(self.packed_padded[idx].cuda())
#        else:
#            return(self.packed_padded[idx])
#
#
#
#
#
#
#
#
#
#file_name='data/resmed/train/train_resmed.pt'
#batch_size = 64
#
#respiration_data = torch.load(file_name)
#batch_size = batch_size
#
## cut length that it matches with batch_size
#
#respiration_data = torch.cat(respiration_data, dim=0)
#num_samples = respiration_data.shape[0]
#num_samples = (num_samples // batch_size) * batch_size
#respiration_data = respiration_data[:num_samples, :, :]
