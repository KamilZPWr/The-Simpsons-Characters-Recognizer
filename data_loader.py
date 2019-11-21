import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data(train_path='data/simpsons_train', test_path='data/simpsons_test', batch_size=160, shuffle=True):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0],
                             [1, 1, 1])
        ])

    train_data = datasets.ImageFolder(os.path.join(train_path), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(test_path), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
