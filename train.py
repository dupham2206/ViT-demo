import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_lightning as pl
from ViT_trainer import ViT
import os

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")

# Define transform for training and testing image
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

# Define train, val, test dataset 
train_dataset = MNIST(root=DATASET_PATH, train=True, download=True, transform=train_transforms)
test_dataset = MNIST(root=DATASET_PATH, train=False, download=True, transform=test_transforms)

# SEED
pl.seed_everything(226)

# SPLIT DATASET
train_dataset, val_dataset = data.random_split(train_dataset, [55000, 5000])

# loader
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

# checkpoint path
CHECKPOINT_PATH = os.environ.get(
    "PATH_CHECKPOINT",
    "checkpoint/"
)

# trainer
trainer = pl.Trainer(
    accelerator="gpu",
    gpus=0,
    default_root_dir = os.path.join(CHECKPOINT_PATH, "vit"),
)

# model args
kwargs = {
    "embbed_dim" : 256,
    "hidden_dim" : 512,
    "num_heads" : 8,
    "num_layers" : 16,
    "patch_size" : 4,
    "num_channels" : 1,
    "num_patches" : 64,
    "num_classes" : 10,
    "dropout" : 0.2,
}

model = ViT(kwargs, lr=3e-4)

# 
trainer.fit(model, train_loader, val_loader)
test_result = trainer.test(model, test_dataset, verbose=False)
print("Results: ", test_result)