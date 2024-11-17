from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class NWRD(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()   

    def load_data(self):
        rust_dir = os.path.join(self.root_dir, "rust")
        non_rust_dir = os.path.join(self.root_dir, "non_rust")

        # Load rust images
        for filename in os.listdir(rust_dir):
            filepath = os.path.join(rust_dir, filename)
            self.images.append(filepath)
            self.labels.append(1)

        # Load non-rust images
        for filename in os.listdir(non_rust_dir):
            filepath = os.path.join(non_rust_dir, filename)
            self.images.append(filepath)
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label
