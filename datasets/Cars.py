import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os

CARS_SAVED_PATH = "/share/coecis/open_active/stanford_cars"

class Cars(Dataset):
    def __init__(self, root, train, transform, num_classes=196, start_indx=0, img_type=".jpg"):
        self.transform = transform
        
        if not os.path.exists(root+"/stanford_cars"):
            import shutil
            shutil.copytree(CARS_SAVED_PATH, root+"/stanford_cars")

        if train:
            self.root = os.path.join(root, 'stanford_cars/train')
        else:
            self.root = os.path.join(root, 'stanford_cars/test')

        self.directories = np.sort(os.listdir(self.root))
        self.annotations = []
        self.class_labels = []
        self.classes = [i for i in range(num_classes)]
        self.img_type = img_type
        
        class_label = 0
        print("Loading...")
        print(start_indx)
        for i in range(start_indx, num_classes+start_indx):
            PATH = os.path.join(self.root, self.directories[i])
            if os.path.isdir(PATH):
                for file in os.listdir(PATH):
                    if file.endswith(self.img_type):
                        self.annotations.append(os.path.join(PATH, file))
                        self.class_labels.append(class_label)
                class_label += 1
        print(len(self.annotations))
        print("done!")

        self.targets = self.get_all_labels()

    def __getitem__(self, index):
        img_id = self.annotations[index]
        label = self.class_labels[index]

        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_all_labels(self):
        return list(map(int, self.class_labels))

    def __len__(self):
        return len(self.annotations)
