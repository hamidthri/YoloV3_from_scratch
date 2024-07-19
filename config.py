import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


Dataset="Pascal_VOC"
lr = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0.0005
num_workers = 2
shuffle = True
pin_memory = True
num_epochs = 100
num_classes = 80 # COCO dataset has 80 classes
csv_train_data = "/"
csv_test_data = "/"
img_dir=Dataset + "/images/"
label_dir=Dataset + "/labels"
S = [19, 38, 76] #image size / 32, / 16, / 8
image_size = 608 


ANCHORS=[
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], #for the first scale
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], #for the second scale
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], #for the third scale
] 

scale_tr = 1.2
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size * scale_tr),
        A.PadIfNeeded(min_height=image_size * scale_tr, min_width=image_size * scale_tr, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(width=image_size, height=image_size),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1, always_apply=False, p=0.5),
        A.OneOf(
            [
                A.ShiftScaleRotate(rotate_limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
                A.IAAAffine(shear=25, p=0.3, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.2),
        A.Blur(p=0.2),
        A.CLAHE(p=0.2),
        A.Posterize(p=0.2),
        A.ToGray(p=0.2),
        A.ChannelShuffle(p=0.1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
        ToTensorV2(),
        
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[""]),
    
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]