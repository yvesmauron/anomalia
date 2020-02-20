import unittest
import pandas as pd
import numpy as np
import torch
import atemteurer.utils as utils
import torch


class SlidingWindowsTest(unittest.TestCase):

    def test_sliding_input_duplicates(self):
        # tests if functions creates duplicate input samples
        test_tensor = torch.zeros(4, 2)
        test_tensor[:, 0] = 1
        test_tensor[:, 1] = 2

        output_tensor = utils.generate_lstm_input_sequence(
            input_tensor=test_tensor,
            seq_len=2,
            window_shift_step_size=2
        )

        output_expected = np.array(
            [[[1., 2.],
              [1., 2.]],
     
             [[1., 2.],
              [1., 2.]]]
        )

        np.testing.assert_array_equal(
            output_tensor.numpy(),
            output_expected
        )

    def test_sliding_input_shift(self):
        test_tensor = torch.zeros(4, 2)
        test_tensor[:, 0] = 1
        test_tensor[:, 1] = 2

        output_tensor = utils.generate_lstm_input_sequence(
            input_tensor=test_tensor,
            seq_len=2,
            window_shift_step_size=1
        )

        output_expected = np.array(
            [[[1., 2.],
              [1., 2.]],
     
             [[1., 2.],
              [1., 2.]],
             
             [[1., 2.],
              [1., 2.]]]
        )

        np.testing.assert_array_equal(
            output_tensor.numpy(),
            output_expected
        )

    def test_get_id_bounds(self):
        test_tensor = torch.Tensor(
            [
                -1,
                -1,
                -1,
                1, #3 inclusive
                1,
                1,
                1,
                -1, #7 exclusive
                -1,
                1,
                -1
            ]
        )
        start, end = utils.get_id_bounds(test_tensor, -1)
        self.assertEquals(start, 3)
        self.assertEquals(end, 7)

    def test_pad_sequences(self):
        tensor_list = [
            torch.tensor([
                [
                    1,2,3
                ],
                [
                    2,3,4
                ]
            ]),
            torch.tensor([
                [
                    1,2,3
                ],
                [
                    1,2,3
                ],
                [
                    1,2,3
                ],
                [
                    1,2,3
                ]
            ])
        ]
        
        out_tensor, mask = utils.padding_tensor(tensor_list)

        expected_tensor = np.array(
            [
                [
                    [1, 2, 3],
                    [2, 3, 4],
                    [2, 3, 4],
                    [2, 3, 4]
                ],
                [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]
                ]
            ]
        )

        expected_mask = np.array(
            [
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]
            ]
        )

        np.testing.assert_array_equal(
            out_tensor.numpy(),
            expected_tensor
        )

        np.testing.assert_array_equal(
            mask.numpy(),
            expected_mask
        )


if __name__ == '__main__':
    unittest.main()