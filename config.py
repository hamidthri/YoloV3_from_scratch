Dataset="Pascal_VOC"
lr = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0.0005
num_workers = 2
shuffle = True
pin_memory = True
num_epochs = 100
num_classes = 80
csv_train_data = "/"
csv_test_data = "/"
img_dir=Dataset + "/images/"
label_dir=Dataset + "/labels"

Anchors=[
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

