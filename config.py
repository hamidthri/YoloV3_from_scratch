lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0.0005
num_workers = 2
shuffle = True
pin_memory = True
num_epochs = 100
