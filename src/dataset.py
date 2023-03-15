import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os
from PIL import Image


class VOCDataset(Dataset):

    def __init__(self, annotations: str, images_dir: str, labels_dir: str, S: int = 7, C: int = 20, transform = None):
        self.df_annotations = pd.read_csv(annotations)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.S = S
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.df_annotations)

    def __getitem__(self, item):
        image_path, labels_path = self.df_annotations.loc[item].tolist()
        image_path = os.path.join(self.images_dir, image_path)
        labels_path = os.path.join(self.labels_dir, labels_path)
        df_labels = pd.read_csv(labels_path, sep=' ', names=['class', 'x', 'y', 'width', 'height'])
        boxes = torch.tensor(df_labels.values)
        image = Image.open(image_path)
        if self.transform is not None:
            # image, boxes = self.transform(image, boxes)
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1)
        labels = torch.zeros(self.S, self.S, self.C + 5)
        for box in boxes:
            class_num, x, y, width, height = box.tolist()
            class_num = int(class_num)
            i, j = int(self.S * x), int(self.S * y)
            cell_x, cell_y = self.S * x - i, self.S * y - j
            cell_width, cell_height = self.S * width, self.S * height

            if labels[i, j, 20] == 0:
                labels[i, j, 20] = 1
                labels[i, j, class_num] = 1
                labels[i, j, 21:25] = torch.tensor([cell_x, cell_y, cell_width, cell_height])

        return image, labels
