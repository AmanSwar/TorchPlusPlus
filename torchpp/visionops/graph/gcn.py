import torch
import torch.nn as nn 

import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

class GCN(nn.Module):
    
    def __init__(self , nfeat , nhid , nclass , dropout):
        super(GCN , self).__init__()
        
        self.gc1 = GraphConvolution(nfeat , nhid)
        self.gc2 = GraphConvolution(nhid , nclass)
        self.dropout = dropout
      
    def forward(self , x , adj):
        x = torch.relu(self.gc1(x , adj))
        x = torch.dropout(x , self.dropout , train=self.training)
        x = self.gc2(x , adj)
        return torch.nn.functional.log_softmax(x , dim=1)