import torch

class LUT(torch.nn.Module):
    def __init__(self, lut_dict, dtype='fp16'):
        super(LUT, self).__init__()
        self.x = torch.nn.Parameter(torch.tensor(lut_dict['x'], dtype=torch.float))
        self.a = torch.nn.Parameter(torch.tensor(lut_dict['a'], dtype=torch.float))
        self.b = torch.nn.Parameter(torch.tensor(lut_dict['b'], dtype=torch.float))

        self._cast_type(dtype)
    
    def compute_lut(self, data):
        x, a, b = self.x, self.a, self.b

        dshape = data.shape

        d = data.view(-1, 1)

        index = torch.sum(d > x, dim=-1)

        s, t = a[index], b[index]

        d = d.view(-1)

        lut_output = s * d + t
                                                                            
        return lut_output.view(dshape)

    def _cast_type(self, data_type):
        if data_type == 'fp32':
            return
        elif data_type == 'fp16':
            self.x = self.x.half()
            self.a = self.a.half()
            self.b = self.b.half()
        else :
            raise Exception('This data type is not supported')
