import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class VSLoss(nn.Module):

    def __init__(self, args, reduction='mean', data_percent = None):
        super(VSLoss, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cls_num_list = data_percent
        #print(cls_num_list)

        
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        
        
        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** args.gamma
        temp = temp / np.min(temp)

        iota_list = args.tau * np.log(cls_probs)
        Delta_list = temp ##gamma = 0.15

        self.iota_list = torch.FloatTensor(iota_list).to(device)
        self.Delta_list = torch.FloatTensor(Delta_list).to(device)
        #self.weight = torch.FloatTensor(per_cls_weights).to(device)
        self.reduction = reduction


    def forward(self, inputs, targets):
        output = inputs / self.Delta_list + self.iota_list
        loss = F.cross_entropy(output, targets, weight=None, reduction=self.reduction)
        return loss
