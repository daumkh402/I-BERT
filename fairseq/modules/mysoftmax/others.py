import torch

def IEXP(x, cuda=False):
    i = torch.tensor([2.0])
    if cuda is True :
        i = i.cuda()
    z = x // -torch.log(i)
    p = x + torch.log(i) * z 
    a, b, c = 0.3585, 1.353, 0.344
    L = a * torch.square(p + b) + c
    return L >> z
