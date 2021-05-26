"""
Create on May 23, 2021 
@Stanford CS231 Final Project

Implement R2Plus1D Model with BERT Temporal Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

from .r2plus1d import r2plus1d_34_32_ig65m, r2plus1d_34_32_kinetics, flow_r2plus1d_34_32_ig65m

class rgb_r2plus1d_32f_34(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34, self).__init__()
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.fc_action = nn.Linear(512, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
    def mars_forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class rgb_r2plus1d_16f_34_bert10(nn.Module):
    """Use when input video length is 16"""
    def __init__(self, num_classes, in_channel, length, modelPath=''):
        super(rgb_r2plus1d_16f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.in_channel = in_channel
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        # self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 2 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 2)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample

class rgb_r2plus1d_32f_34_bert10(nn.Module):
    """Use when input video length is 32"""
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        # self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 4 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class rgb_r2plus1d_64f_34_bert10(nn.Module):
    """Use when input video length is 64"""
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        # self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 8)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
if __name__ == "__main__":
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = torch.rand(1, 3, 16, 60, 60).to(device)
    net = rgb_r2plus1d_16f_34_bert10(num_classes=51, length=16)
    net.to(device)
    outputs = net.forward(inputs)
    print(outputs)
    print("Simple Test Passed rgb_r2plus1d_32f_34_bert10") 
