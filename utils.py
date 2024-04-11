from torh.utils.data.Dataloader import DataLoader
from dataset import YoloDataset
import config


def get_data(csv_train_data, csv_test_data):
    
    train_dataset = YoloDataset(
        csv_dir=config.csv_data,
        img_dir=config.img_dir,
        label_dir=config.label_dir,
        anchors=config.anchors,
    )
    
    test_dataset = YoloDataset(
        csv_dir=config.csv_train_data,
        img_dir=config.img_dir,
        label_dir=config.label_dir,
        anchors=config.anchors,
    )
    
    train_loader = Dataloader(
        dataset=config.csv_test_data, 
    )
    
    test_loader
    
    return train_loader, test_loader