import torch
import torch.nn as nn
from lib.utils.utils import list_tensor
from lib.config.para import parse_opt
opt = parse_opt()


def get_center(prototype,component):
    c = []
    for i in prototype:
        c.append(i)
    c = list_tensor(c)
    c = c.view(component*opt.Intent_class,-1)
    return c


def cosinematrix(A):
    prod = torch.mm(A, A.t())#分子
    norm = torch.norm(A,p=2,dim=1).unsqueeze(0)#分母
    cos = prod.div(torch.mm(norm.t(),norm))
    return cos


def diversity_loss(pro):
    dis = cosinematrix(pro)
    div_loss = torch.norm(dis,'fro')
    return div_loss


def calculate_cls(center):
    part = center.chunk(opt.Intent_class, dim=0)
    for j,i in enumerate(part):
        m = torch.mean(i,dim=0).view(1,-1)
        if j==0:
            prot = torch.cat([m],dim=0)
        else:
            prot = torch.cat([prot,m],dim=0)
    return prot


class Mydata(nn.Module):
    def __init__(self,data):
        super(Mydata, self).__init__()
        self.data = torch.nn.Parameter(data,True)

    def update(self,data):
        self.data = torch.nn.Parameter(data,True)

    def forward(self):
        return self.data