import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import iou

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid error: "image file is truncated"

class YoloDataset(Dataset):
    def __init__(self,
                 csv_dir,
                 img_dir,
                 label_dir,
                 anchors, # 3 anchors for 3 scales: anchors sample: [[0.275, 0.3203125], [0.068, 0.11328125], [0.017, 0.03]] 
                 image_size=416,
                 S=[19, 38, 76],
                 C=80, 
                 transform=None):
        self.annotations = pd.read_csv(csv_dir)
        self.image_dir = img_dir
        self.labels = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # 3 anchors for 3 scales
        self.S = S
        self.num_anchors = self.anchors.shape[0] # 9 anchors.shape = (9, 2)
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_threshold = 0.5
        
        def __len__(self):
            return len(self.annotations) # number of samples in dataset
        
        def __getitem__(self, index): # retrieve a sample from dataset
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4).tolist() # [class, x, y, w, h] -> [x, y, w, h, class]
            img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
            image = np.array(Image.open(img_path).convert("RGB"))
            
            if self.transform:
                augmentations = self.transform(image=image, bboxes=bboxes)
                image = augmentations["image"]
                bboxes = augmentations["bboxes"]
            
            targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 = [P_o, x, y, w, h, class=80] P_o = 1 if object exists in cell
            
            for box in bboxes:
                iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # iou of ground truth box with anchors num = 9
                anchor_indices = iou_anchors.argsort(descending=True, dim=0)
                x, y, w, h, class_label = box
                has_anchor = [False, False, False] 
                
                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2 which target we are going to assign
                    anchor_on_scale = anchor_idx % self.num_anchors_per_scale # 0, 1, 2 which anchor on scale we are going to assign
                    S = self.S[scale_idx]
                    i, j = int(S * y), int(S * x)
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # 0 = P_o
                    if not anchor_taken and not has_anchor[scale_idx]: 
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x - j, S * y - i # x and y relative to cell and x_cell, y_cell = 0-1
                        width_cell, height_cell = w * S, h * S 
                        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True
                        
                    elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction example: no object but iou > threshold 
                        
        return image, tuple(targets)
