import torch
import numpy as np
from .nn_model import ApproximatorNN
from .lut import LUT
from .others import *

class MyGelu(torch.nn.Module):
    def __init__(self, run_type = 'nn', gelu_dict = None, dtype='fp32'):
        super(MyGelu, self).__init__()
        self.run_type = run_type

        if run_type == 'nn' and (gelu_dict is None) :
            raise Exception('Please enter the proper data file')

        if self.run_type == 'nn':
            self.n_neuron_gelu = gelu_dict['n_neurons']
            self.nn_gelu =  ApproximatorNN(gelu_dict['n_neurons'])
            self.nn_gelu.load_state_dict(gelu_dict['state_dict'])


        #    if dtype == 'fp16':
        #        self.nn_exp.half()
        #        self.nn_div.half()

        elif self.run_type == 'lut':     
            self.lut_gelu = LUT(gelu_dict, dtype)
        elif self.run_type == 'ibert':
            pass
        else :
            raise Exception('Please enter the proper softmax type')

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
        # transformer has 4-d tensor
        # and softmax is applied through last dim

        gelu_output = self.nn_gelu(x)

        return gelu_output

    def _forward_lut(self, x):
        # transformer has 4-d tensor
        # and softmax is applied through last dim

        gelu_output = self.lut_gelu.compute_lut(x)

        return gelu_output

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
