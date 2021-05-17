import torch
import numpy as np
from .nn_model import ApproximatorNN
from .lut import LUT
from .others import *

class MYLayernorm(torch.nn.Module):
    def __init__(self, run_type = 'nn', sqrt_dict = None, dtype='fp32'):
        super(MYLayernorm, self).__init__()
        self.run_type = run_type

        if run_type == 'nn' and (sqrt_dict is None) :
            raise Exception('Please enter the proper data file')

        if self.run_type == 'nn':
            self.n_neuron_sqrt = sqrt_dict['n_neurons']
            self.nn_sqrt=  ApproximatorNN(sqrt_dict['n_neurons'])
            self.nn_sqrt.load_state_dict(sqrt_dict['state_dict'])


        #    if dtype == 'fp16':
        #        self.nn_exp.half()
        #        self.nn_div.half()

        elif self.run_type == 'lut':     
            self.lut_sqrt = LUT(sqrt_dict, dtype)
        elif self.run_type == 'ibert':
            pass
        else :
            raise Exception('Please enter the proper softmax type')

        self.eps = 1e-5


    def forward(self, x):
        if self.run_type == 'nn':
            return self._forward_nn(x)
        elif self.run_type == 'lut':
            return self._forward_lut(x)
        elif self.run_type == 'ibert':
            return self._forward_ibert(x)
        else:
            raise Exception('This softmax type is not supported')

    def _forward_nn(self, x):
        mean = x.mean(axis=2, keepdim=True)
        y = x - mean
        var =  torch.mean(y ** 2, axis=2, keepdim=True)

        # x = y / torch.sqrt(self.eps + var)

        sqrt = self.nn_sqrt(self.eps + var)
        x = y / sqrt

        return x

    def _forward_lut(self, x):
        mean = x.mean(axis=2, keepdim=True)
        y = x - mean
        var =  torch.mean(y ** 2, axis=2, keepdim=True)

        sqrt = self.lut_sqrt.compute_lut(self.eps + var)
        x = y / sqrt

        return x

    def _forward_ibert(self, x):
        # transformer has 4-d tensor
        # and softmax is applied through last dim
        d1, d2, d3, d4 = x.shape
        max_values = torch.max(x, dim=-1, keepdim=True).values
        in_values = x - max_values

        iexp_output = IEXP(in_values.view(-1,1), cuda=True)
        iexp_output = iexp_output.view(-1, d4)

        softmax_output = iexp_output / torch.sum(iexp_output, dim=-1, keepdim=True)
        softmax_output = softmax_output.view(d1, d2, d3, d4)

        return softmax_output
