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


def IGELU(x, cuda=False):

    a, b = -0.2888, -1.769

    sign = 1 if x > 0 else -1 if a < 0 else 0

    L = sign * ( a * torch.square( 
                    (torch.clamp( torch.abs(x / torch.sqrt(2.0)), max = -b) + b)) \
                + 1)
         
    out = x * 0.5 * (1 + L)



    