import json
import os
from collections import namedtuple
import cv2

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Satellites_building(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on sia
    SatellitesClass = namedtuple('SatellitesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        SatellitesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        SatellitesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        SatellitesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        SatellitesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        SatellitesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        SatellitesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        SatellitesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        SatellitesClass('road',                 7, 255, 'flat', 1, False, False, (128, 64, 128)),
        SatellitesClass('sidewalk',             8, 255, 'flat', 1, False, False, (244, 35, 232)),
        SatellitesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        SatellitesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        SatellitesClass('building',             11, 1, 'construction', 1, True, False, (70, 70, 70)),
        SatellitesClass('wall',                 12, 255, 'construction', 2, False, False, (102, 102, 156)),
        SatellitesClass('fence',                13, 255, 'construction', 2, False, False, (190, 153, 153)),
        SatellitesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        SatellitesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        SatellitesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        SatellitesClass('pole',                 17, 255, 'object', 3, False, False, (153, 153, 153)),
        SatellitesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        SatellitesClass('traffic light',        19, 255, 'object', 3, False, False, (250, 170, 30)),
        SatellitesClass('traffic sign',         20, 255, 'object', 3, False, False, (220, 220, 0)),
        SatellitesClass('vegetation',           21, 255, 'nature', 4, False, False, (107, 142, 35)),
        SatellitesClass('terrain',              22, 255, 'nature', 4, False, False, (152, 251, 152)),
        SatellitesClass('sky',                  23, 255, 'sky', 5, False, False, (70, 130, 180)),
        SatellitesClass('person',               24, 255, 'human', 6, False, False, (220, 20, 60)),
        SatellitesClass('rider',                25, 255, 'human', 6, False, False, (255, 0, 0)),
        SatellitesClass('car',                  26, 255, 'vehicle', 7, False, False, (0, 0, 142)),
        SatellitesClass('truck',                27, 255, 'vehicle', 7, False, False, (0, 0, 70)),
        SatellitesClass('bus',                  28, 255, 'vehicle', 7, False, False, (0, 60, 100)),
        SatellitesClass('caravan',              29, 255, 'vehicle', 7, False, True, (0, 0, 90)),
        SatellitesClass('trailer',              30, 255, 'vehicle', 7, False, True, (0, 0, 110)),
        SatellitesClass('train',                31, 255, 'vehicle', 7, False, False, (0, 80, 100)),
        SatellitesClass('motorcycle',           32, 255, 'vehicle', 7, False, False, (0, 0, 230)),
        SatellitesClass('bicycle',              33, 255, 'vehicle', 7, False, False, (119, 11, 32)),
        SatellitesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='sia', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}{}'.format(file_name.split('.')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 1
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    def make_encode_target(self, target):
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(target, 20, 1, cv2.THRESH_BINARY)
        return dst

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        
        if self.transform:
            image, target = self.transform(image, target)
        target = np.array(target)
        target = self.make_encode_target(target)
        target = np.array(target, dtype='int')
        
        return image, target

    def __getimg__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        print(self.images[index])
        #target = Image.open(self.targets[index])
        #if self.transform:
        #    image, target = self.transform(image, target)
        #target = self.encode_target(target)
        return image #, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
        elif target_type == 'sia':
            return '.png'
