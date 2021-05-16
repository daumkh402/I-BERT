import torch
import numpy as np

class ApproximatorNN(torch.nn.Module):
    def __init__(self, n_neurons, init_type = 'normal'):
        super(ApproximatorNN, self).__init__()
        self.n_neurons = n_neurons
        self.fc1 = torch.nn.Linear(1, self.n_neurons, bias=False)
        self.fc2 = torch.nn.Linear(self.n_neurons, 1)
        self.relu = torch.nn.ReLU(inplace=True)
                                                            
        if init_type == 'normal' :
            # normalize the inflection point
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            self.fc1.weight.data = torch.abs(self.fc1.weight.data)*self.n_neurons/2
        elif init_type == 'div' :
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            self.fc1.weight.data = -torch.abs(self.fc1.weight.data)
        else :
            raise Exception('Please check the initialization type') 

    def forward(self, x):
        # forward function, + 1 indicates fixed bias value
        x = self.relu(self.fc1(x) + 1)
        x = self.fc2(x)
        return x
    
    def lut_gen(self):
        # prepare the weight/bias data
        l = self.fc1.weight.shape[0]
    
        fc1_weight = self.fc1.weight.data.view(-1).cpu().numpy()
        fc2_weight = self.fc2.weight.data.view(-1).cpu().numpy()
    
        if self.fc1.bias is not None :
            fc1_bias = self.fc1.bias.data.view(-1).cpu().numpy()
        else : 
            fc1_bias = np.ones(l)
    
        if self.fc2.bias is not None :
            fc2_bias = self.fc2.bias.data.view(-1).cpu().numpy()[0]
        else : 
            fc2_bias = 0 
    
        # compute inflection point and intermediate scale and bias
        x_pos = []
        m = []
        n = []
        for i in range(l):
            x_pos += [fc1_bias[i] / fc1_weight[i] * -1]
            m += [fc1_weight[i] * fc2_weight[i]]
            n += [fc1_bias[i] * fc2_weight[i]]
    
        idx = np.argsort(x_pos)
        x_pos = np.array(x_pos)[idx]
        m = np.array(m)[idx]
        n = np.array(n)[idx]
        w = np.array(fc1_weight)[idx]
    
        # compute final scale and bias
        on = w < 0 
        a = [sum(m[on])]
        b = [sum(n[on])]
        for i, x in enumerate(x_pos) :
            if w[i] < 0 :
                on[i] = 0
            else :
                on[i] = 1
            a += [sum(m[on])]
            b += [sum(n[on]) + fc2_bias]
            
        return x_pos, a, b

