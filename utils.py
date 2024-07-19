from torh.utils.data.Dataloader import DataLoader
from dataset import YoloDataset
import config


def get_data(csv_train_data, csv_test_data):
    
    train_dataset = YoloDataset(
        csv_dir=config.csv_data,
        transform=config.train_transforms,
        img_dir=config.img_dir,
        S=config.S,
        label_dir=config.label_dir,
        anchors=config.ANCHORS,
    )
    
    test_dataset = YoloDataset(
        csv_dir=config.csv_train_data,
        transform=config.test_transforms,
        img_dir=config.img_dir,
        S=config.S,
        label_dir=config.label_dir,
        anchors=config.ANCHORS,
    )
    
    train_loader = Dataloader(
        dataset=config.csv_train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    test_loader = Dataloader(
        dataset=config.csv_test_data,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    train_eval_dataset = YoloDataset(
        csv_dir=config.csv_train_data,
        transform=config.test_transforms,
        img_dir=config.img_dir,
        S=config.S,
        label_dir=config.label_dir,
        anchors=config.ANCHORS,
    )
    
    train_eval_loader = Dataloader(
        dataset=config.csv_train_data,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    return train_loader, test_loader, train_eval_loader