import config
import torch
import torch.optim as optim

from model import YoloV3
from tqdm import tqdm

from loss import YoloLoss


torch.backends.cudnn.benchmark = True

def train(train_loader, model, optimizer, loss_fn):
    loop  = tqdm(train_loader, leave=True)
    losses = []
    
    for batch_idx, (x, y)


if __name__ == "__main__":