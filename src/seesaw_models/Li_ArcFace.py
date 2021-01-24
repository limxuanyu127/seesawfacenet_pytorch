import torch
from torch import nn
import torch.nn.functional as F
from math import pi
from torch.nn import Parameter

class LiArcFace(nn.Module):
    def __init__(self, embedding_size=512, classnum=51332, s=64.0,  m=0.42):
        super().__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        self.weight = nn.Parameter(torch.empty(classnum, embedding_size))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s

    def forward(self, input, label):
        W = F.normalize(self.weight)
        input = F.normalize(input)
        cosine = input @ W.t()
        theta = torch.acos(cosine)
        m = torch.zeros_like(theta)
        m.scatter_(1, label.view(-1, 1), self.m)
        logits = self.s * (pi - 2 * (theta + m)) / pi
        return logits