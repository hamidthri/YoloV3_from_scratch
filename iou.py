import torch
import numpy as np

def iou(box1, box2):
    
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    
    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area_box2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union = area_box1 + area_box2 - intersection
    iou = intersection / union
    
    return iou
    