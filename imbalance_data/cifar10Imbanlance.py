import torchvision
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class Cifar10Imbalance(Dataset):
    def __init__(self, imbanlance_rate, num_cls=10, file_path="data/",
                 train=True, transform=None, label_align=True, category_image_counts=None):
        self.transform = transform
        self.label_align = label_align
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must be between 0.0 and 1"
        self.imbanlance_rate = imbanlance_rate
        self.category_image_counts = category_image_counts

        self.num_cls = num_cls
        self.data, _ = self.produce_imbalance_data(file_path=file_path, train=train, imbanlance_rate=self.imbanlance_rate)
        self.x = self.data['x']
        self.targets = self.data['y'].tolist()
        self.y = self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.class_list

    def produce_imbalance_data(self, imbanlance_rate, file_path="/data", train=True):
        train_data = torchvision.datasets.CIFAR10(
            root=file_path,
            train=train,
            download=True,
        )
        x_train = train_data.data
        y_train = train_data.targets
        y_train = np.array(y_train)

        if self.category_image_counts is None:
            data_percent = []
            for cls_idx in range(self.num_cls):
                if train:
                    num = int(len(x_train) / self.num_cls * (imbanlance_rate ** (cls_idx / (self.num_cls - 1))))
                else:
                    num = int(len(x_train) / self.num_cls)
                data_percent.append(num)
        else:
            data_percent = self.category_image_counts

        self.class_list = data_percent

        rehearsal_data = None
        rehearsal_label = None
        for i in range(self.num_cls):
            index = (y_train == i)
            task_train_x = x_train[index]
            label = y_train[index]

            data_num = len(task_train_x)
            selected_num = min(data_percent[i], data_num)
            index = np.random.choice(data_num, selected_num, replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        return task_split, data_percent
