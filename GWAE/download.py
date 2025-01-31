import os
from torchvision import datasets

# Define the root directory where you want to store the dataset
root_dir = './data'  # You can specify your custom directory

# Download the CelebA dataset
celeba_dataset = datasets.CelebA(root=root_dir, download=True, split='train')

# Check if the dataset is downloaded
print(f"Dataset downloaded to: {root_dir}")
print(f"Total samples in training set: {len(celeba_dataset)}")
