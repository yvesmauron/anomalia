import os
# data manipulating
import numpy as np
import pandas as pd
import math
# logging
import logging
import logging.config
import torch
from torch import Tensor
# logging
from tqdm import tqdm

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('anomalia')


def generate_lstm_input_sequence(
    input_tensor: Tensor,
    seq_len: int,
    window_shift_step_size: int
):
    """LSTM input

    Generates a tensor that corresponds an LSTM input sequence from a 
    two dimensional table (rows = samples, columns = variables)

    Arguments:
        input_tensor (Tensor): the tensor that should be shifted
        seq_len (int): the length the tensor should be 
        window_shift (int): how much time_steps the input window should be shifted

    Returns:
        Tensor: reshaped tensor
    """
    num_iterations = (seq_len // window_shift_step_size)
    num_vars = input_tensor.shape[1]
    tensor_list = []
    for i in range(num_iterations):
        # calculate how much the window has to be shifted
        window_shift = i * window_shift_step_size
        # shift the input tensor
        shifted_tensor = input_tensor[window_shift:, :]
        # evaluate new size
        total_time_steps = shifted_tensor.shape[0]
        # evalute the new sample size
        sample_size = total_time_steps // seq_len
        # crop samples that cannot be used (as not devidable by sample size)
        upper_bound = sample_size * seq_len
        # log values
        logger.debug('creating {} samples using data idx {} to {}'.format(
            str(sample_size),
            str(window_shift),
            str(upper_bound + window_shift)
        ))
        # subset shifted tensor to match sample size
        subset_tensor = shifted_tensor[0:upper_bound, :]
        # create input_samples
        input_samples = subset_tensor.view(sample_size, seq_len, num_vars)
        # add it to the list
        tensor_list.append(input_samples)

    return(torch.cat(tensor_list, dim=0))


def get_id_bounds(
    values: Tensor,
    default_value: float
):
    """Gets the maximum sequence bounds of non idle time

    Machines shows default values at the beginning and end of the
    operations; this functions returns the ids of the longest sequence
    that is not operating with the default values. 

    Note that you cannot just remove all default values, essentially
    because order matters and there might be also intermediate interuptions.
    Just removing this kind of outliers in between the opertions would lead to
    wrong time series (time-gaps).

    Arguments:
        values (Tensor): 1d torch tensor
        max (float): default value the machine is operating in

    Returns:
        int, int: start and end of sequence
    """
    # get all values that are not default ones
    default_value_idx = (values == default_value).nonzero()[:, 0]
    # get the longest sequence without interruption
    # to do this, get the difference of the above ids
    diff = default_value_idx[1:] - default_value_idx[:-1]
    # find the maximum difference (maximum ids between default values)
    split = (diff == diff.max()).nonzero()[0, 0]
    # return start, end ids
    start = default_value_idx[split] + \
        1 if split != 0 and diff.max() != 1 else 0
    end = default_value_idx[split +
                            1] if split != 0 and diff.max() != 1 else default_value_idx[0]
    return start, end


def padding_tensor(sequences, max_length=1000000):
    """Pad sequences that are too short

    Arguments:
        sequences (list(Tensor)): list of tensors
        max_length (int): maximum supported length of a breath

    Returns:
        (max_length, out_tensor, mask): int, padded tensor, mask
    """
    # get the number of sequences
    num = len(sequences)
    # get the maximum length (clip too long sequences)
    max_len = min(max([s.shape[0] for s in sequences]), max_length)
    # define new output dimensions
    out_dims = (num, max_len, *sequences[0].shape[1:])
    # create output_tensor with new dimensionality
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    # create new mask_tensor with the corresponding mask
    mask = sequences[0].data.new(*out_dims).fill_(0)
    # iterate over the sequences
    logger.info('Start padding breaths....')
    with tqdm(total=len(sequences), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
        for i, tensor in enumerate(sequences):
            # get the length of the current breath
            length = min(tensor.size(0), max_len)
            # add all valid breaths
            print(tensor)
            input('before')
            out_tensor[i, :length] = tensor[:length, :]
            # for the breaths that are "too short" padd with last value
            out_tensor[i, length:] = 0
            print(out_tensor)
            input('after')
            # create mask
            mask[i, :length] = 1
            # update progressbar
            pbar.update(1)

    # return result
    return max_len, out_tensor, mask
