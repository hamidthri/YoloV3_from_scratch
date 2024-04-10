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
        class_loss  = self.entropy(
            (predictions[..., 5,:][obj]),(target[..., 5][obj]).long() # 0:5 P_o, x, y, w, h, class 5:80 
        )
        
        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3: 3 anchors, 2: x, y reshape: because of broadcasting
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1 # 1:3 x, y, 3:5 w, h
        )
        ious = iou(box_preds[obj], target[..., 1:5][obj], box_format="midpoint").detach()
        obj_loss = self.bce(
            predictions[..., 0:1][obj], ious * target[..., 0:1][obj]
        )
        
        # no object loss
        noobj_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )
        
        # box coordinate loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )
        box_loss = self.mse(
            predictions[..., 1:5][obj], target[..., 1:5][obj]
        )
        
        return(
            self.lambda_box * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )
        