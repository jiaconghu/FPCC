from torch.utils.data import DataLoader
from torchvision import transforms
from loaders.datasets import ImageDataset

mnist_train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

mnist_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768)),
])

cifar10_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768)),
])


def _get_set(data_path, transform):
    return ImageDataset(image_dir=data_path,
                        transform=transform)


def load_images(data_dir, data_name, data_type=None):
    assert data_name in ['mnist', 'fashion-mnist', 'cifar10', 'cifar100']
    assert data_type is None or data_type in ['train', 'test']

    data_transform = None
    if data_name == 'mnist' and data_type == 'train':
        data_transform = mnist_train_transform
    elif data_name == 'mnist' and data_type == 'test':
        data_transform = mnist_test_transform
    elif data_name == 'cifar10' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar10' and data_type == 'test':
        data_transform = cifar10_test_transform
    elif data_name == 'cifar100' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar100' and data_type == 'test':
        data_transform = cifar10_test_transform
    assert data_transform is not None

    data_set = _get_set(data_dir, transform=data_transform)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=256,
                             num_workers=4,
                             shuffle=True)
    return data_loader
