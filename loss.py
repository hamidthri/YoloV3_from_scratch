import torch
import torch.nn as nn

import iou


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        # Constants
        self.lambda_class = 1 
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
    
    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        
        # Class loss
        