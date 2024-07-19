import config
import torch
import tocch.utils.data.Dataloader as DataLoader
import torch.optim as optim

from model import YoloV3
from tqdm import tqdm

from loss import YoloLoss
import utils


torch.backends.cudnn.benchmark = True

def train(train_loader, model, optimizer, loss_fn, scaled_anchors, scaler):
    loop  = tqdm(train_loader, leave=True)
    losses = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        with torch.cuda.amp.autocast():
            out = model(x) 
            loss = (loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2]))
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():
    model = YoloV3(num_classes=config.num_classes).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, test_loader, train_eval_loader = utils.get_data(config.csv_train_data/train.csv, config.csv_test_data/test.csv)
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)
    for epoch in range(cofig.Epochs):
        train(train_loader, model, optimizer, loss_fn, scaled_anchors, scaler)
        
        if config.save_model:
            torch.save(model.state_dict(), "yolov3.pth")
        
        if epoch % 10 == 0 and epoch > 0:
            mean_avg_prec = get_mean_avg_prec(model, test_loader)
            print(f"Mean average precision at epoch {epoch} is {mean_avg_prec}")
        
    
if __name__ == "__main__":
    main()