import torchvision.transforms as transforms
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import csv
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import math


def encode_loc_time(loc_time):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    feats = np.concatenate((np.sin(math.pi * loc_time), np.cos(math.pi * loc_time)))
    return feats


class Lmaster_train(Dataset):

    def __init__(self, path, File_path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.year = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        self.avg = []
        for item in reader:
            image_path0 = os.path.join(path, 'Lmaster_images', item[0])
            image_path1 = os.path.join(path, 'u', item[0])
            self.images_path.append(image_path0)
            self.images_path1.append(image_path1)
            self.train_id.append(int(item[1]))

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]

        image_path1 = self.images_path1[index]

        image = default_loader(image_path)
        image1 = default_loader(image_path1)
        i = cv2.imread(image_path1, 0)
        s = np.sum(i)
        l = len(np.flatnonzero(i))
        a = s / l / 255
        av = [a for i in range(30)]
        av = torch.tensor(av).data.float()
        av = encode_loc_time(av)

        u = torch.tensor([0, 1, 2, 3, 4, 5])
        avg = torch.tensor([a]).data.float().expand(1, 224, 224)
        average = 0
        if self.transform:
            image = self.transform(image)
            image1 = self.transform(image1)
            imgs = torch.cat((image[0].view(1, 224, 224), image1[0].view(1, 224, 224), avg), 0)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id, average, image1, imgs, av


class Lmaster_test(Dataset):

    def __init__(self, path, File_path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.imagename = []
        self.year = []
        self.avg = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        for item in reader:
            image_path0 = os.path.join(path, 'Lmaster_images', item[0])
            image_path1 = os.path.join(path, 'u', item[0])
            self.images_path1.append(image_path1)
            self.images_path.append(image_path0)
            self.train_id.append(int(item[1]))
            self.imagename.append(item[0])

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]
        image_path1 = self.images_path1[index]
        imagename = self.imagename[index]
        image = default_loader(image_path)
        image1 = default_loader(image_path1)
        i = cv2.imread(image_path1, 0)
        s = np.sum(i)
        l = len(np.flatnonzero(i))
        a = s / l / 255
        av = [a for i in range(30)]
        av = torch.tensor(av).data.float()
        av = encode_loc_time(av)
        avg = torch.tensor([a]).data.float().expand(1, 224, 224)
        average = 0
        if self.transform:
            image = self.transform(image)
            image1 = self.transform(image1)
            imgs = torch.cat((image[0].view(1, 224, 224), image1[0].view(1, 224, 224), avg), 0)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id, imagename, image_path, image1, imgs, av


class Lmaster_val(Dataset):

    def __init__(self, path, File_path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.imagename = []
        self.year = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        for item in reader:
            image_path0 = os.path.join(path, 'Lmaster_images', item[0])
            image_path1 = os.path.join(path, 'u', item[0])
            self.images_path1.append(image_path1)
            self.images_path.append(image_path0)
            self.train_id.append(int(item[1]))

            self.imagename.append(item[0])

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]
        image_path1 = self.images_path1[index]
        imagename = self.imagename[index]
        image = default_loader(image_path)
        image1 = default_loader(image_path1)
        i = cv2.imread(image_path1, 0)
        s = np.sum(i)
        l = len(np.flatnonzero(i))
        a = s / l / 255
        av = [a for i in range(30)]
        av = torch.tensor(av).data.float()
        av = encode_loc_time(av)
        avg = torch.tensor([a]).data.float().expand(1, 224, 224)
        img = image
        u = torch.tensor([0, 1, 2, 3, 4, 5])
        if self.transform:
            image = self.transform(image)
            image1 = self.transform(image1)
            imgs = torch.cat((image[0].view(1, 224, 224), image1[0].view(1, 224, 224), avg), 0)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id, imagename, image_path, image1, imgs, av


class Lmaster_val1(Dataset):

    def __init__(self, path, File_path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.imagename = []
        self.year = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        self.num = []
        for item in reader:
            image_path0 = os.path.join(path, 'Lmaster_images', item[0])
            image_path1 = os.path.join(path, 'u', item[0])
            self.images_path.append(image_path0)
            self.images_path1.append(image_path1)
            self.train_id.append(int(item[1]))
            self.imagename.append(item[0])
            self.num.append(int(item[2]))

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]
        image_path1 = self.images_path1[index]
        imagename = self.imagename[index]
        number = self.num[index]
        # year=self.year[index]
        image = default_loader(image_path)
        image1 = default_loader(image_path1)
        i = cv2.imread(image_path1, 0)
        s = np.sum(i)
        l = len(np.flatnonzero(i))
        a = s / l / 255
        avg = torch.tensor([a]).data.float().expand(1, 224, 224)
        img = image
        if self.transform:
            image = self.transform(image)
            image1 = self.transform(image1)
            imgs = torch.cat((image[0].view(1, 224, 224), image1[0].view(1, 224, 224), avg), 0)
            # print(image.shape)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id, imagename, image_path, number, imgs, image1, image_path1


class as_oct(Dataset):

    def __init__(self, root, File_path, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.year = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        self.avg = []
        for item in reader:
            image_name = item[0]
            image_path = os.path.join(self.root, image_name)
            self.images_path.append(image_path)
            self.train_id.append(int(item[1]))

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id


class as_oct1(Dataset):

    def __init__(self, root, File_path, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = []
        self.images_path1 = []
        self.year = []
        csvFile = open(File_path, "r")
        reader = csv.reader(csvFile)
        self.train_id = []
        self.avg = []
        self.num = []
        self.name = []
        for item in reader:
            image_name = item[0]
            image_path = os.path.join(self.root, image_name)
            self.name.append(image_name)
            self.images_path.append(image_path)
            self.train_id.append(int(item[1]))
            self.num.append(int(item[2]))

    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        class_id = self.train_id[index]
        image_path = self.images_path[index]
        image = default_loader(image_path)
        num = self.num[index]
        name = self.name[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id, name, num


if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    data_loader = DataLoader(Lmaster_train(path='/media/xxx/3AF0749EF07461D5/FLX',
                                           File_path='/media/xxx/3AF0749EF07461D5/FLX/123456.csv',
                                           transform=train_transforms, target_transform=None),
                             batch_size=1,
                             shuffle=False,
                             num_workers=16)
    for i, data in enumerate(data_loader):
        input, labels, average, lens, v = data
