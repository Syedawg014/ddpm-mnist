import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=128, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,train_dataset