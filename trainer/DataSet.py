import torch
from skimage import io
import torchvision.transforms as transforms

import os


class DroneImagesDataSet(torch.utils.data.Dataset):
    """Drone Images dataset"""

    def __init__(self, labels_path, root_dir):
        """
        Args:
            labels_path (string): Path to the labels file.
            root_dir (string): Directory with all the images.
        """
        self.labels_path = labels_path
        self.root_dir = root_dir

        with open(labels_path, 'r') as fd:
        	labels = fd.read()
        	labels = labels.split()

        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx].split(";")
        img_name = os.path.join(self.root_dir, label[0]+".jpg")
        image = io.imread(img_name)
        image = self.transform(image)
        label = [float(elt) for elt in label]
        label = torch.FloatTensor(label[1::])
        sample = {'image': image, 'label': label}

        return sample