import torch
import numpy as np
from .nn_model import ApproximatorNN
from .lut import LUT
from .others import *

class MySoftmax(torch.nn.Module):
    def __init__(self, run_type = 'nn', exp_dict = None, div_dict = None, dtype='fp32'):
        super(MySoftmax, self).__init__()
        self.run_type = run_type

        if run_type == 'nn' and (exp_dict is None or div_dict is None) :
            raise Exception('Please enter the proper data file')

        if self.run_type == 'nn':
            self.n_neuron_exp = exp_dict['n_neurons']
            self.nn_exp = ApproximatorNN(exp_dict['n_neurons'])
            self.nn_exp.load_state_dict(exp_dict['state_dict'])

            self.n_neuron_div = div_dict['n_neurons']
            self.nn_div = ApproximatorNN(div_dict['n_neurons'])
            self.nn_div.load_state_dict(div_dict['state_dict'])
            
        #    self.nn_exp.cuda()
        #    self.nn_div.cuda()

        #    if dtype == 'fp16':
        #        self.nn_exp.half()
        #        self.nn_div.half()

        elif self.run_type == 'lut':
            self.lut_exp = LUT(exp_dict, dtype)
            self.lut_div = LUT(div_dict, dtype)

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
        d1, d2, d3, d4 = x.shape

        max_values = torch.max(x, dim=-1, keepdim=True).values
        in_values = x - max_values

        exp_output = self.nn_exp(in_values.view(-1,1))
        #exp_output = torch.exp(in_values.view(-1,1))
        exp_output = exp_output.view(-1, d4)

        exp_sum = torch.sum(exp_output, dim=-1, keepdim=True)

        softmax_output = exp_output * self.nn_div(exp_sum)
        #softmax_output = exp_output / exp_sum
        softmax_output = softmax_output.view(d1, d2, d3, d4)

        return softmax_output

    def _forward_lut(self, x):
        # transformer has 4-d tensor
        # and softmax is applied through last dim
        d1, d2, d3, d4 = x.shape

        max_values = torch.max(x, dim=-1, keepdim=True).values
        in_values = x - max_values

        exp_output = self.lut_exp.compute_lut(in_values)

        ## test 
        exp_output = exp_output * (x > -9990) 
        
        exp_output = exp_output.view(-1, d4)

        exp_sum = torch.sum(exp_output, dim=-1, keepdim=True)

        softmax_output = exp_output * self.lut_div.compute_lut(exp_sum)
        softmax_output = softmax_output.view(d1, d2, d3, d4)

        return softmax_output

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
