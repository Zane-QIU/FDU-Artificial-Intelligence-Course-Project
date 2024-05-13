import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class HandwrittenChineseDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label in range(1, 13):
            folder_path = os.path.join(root_dir, str(label))
            images = os.listdir(folder_path)
            for img in images:
                self.samples.append((os.path.join(folder_path, img), label-1))

        # Split into train and validation sets
        self.train_samples, self.val_samples = train_test_split(self.samples, test_size=0.2, random_state=42)
        self.samples = self.train_samples if train else self.val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label
