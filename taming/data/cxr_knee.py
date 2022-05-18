import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
import albumentations

from torch.utils.data import Dataset


def rgba_to_depth(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    return np.ascontiguousarray(y)


def get_preprocessor(size=None, random_crop=False, additional_targets=None,
                     crop_size=None):
    if size is not None and size > 0:
        transforms = list()
        rescaler = albumentations.SmallestMaxSize(max_size = size)
        transforms.append(rescaler)
        if not random_crop:
            cropper = albumentations.CenterCrop(height=size,width=size)
            transforms.append(cropper)
        else:
            cropper = albumentations.RandomCrop(height=size,width=size)
            transforms.append(cropper)
            flipper = albumentations.HorizontalFlip()
            transforms.append(flipper)
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    elif crop_size is not None and crop_size > 0:
        if not random_crop:
            cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
        else:
            cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
        transforms = [cropper]
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    else:
        preprocessor = lambda **kwargs: kwargs
    return preprocessor


class BaseWithPrevious(Dataset):
    def __init__(self, config=None, csv_path=None, size=None, random_crop=False,
                 crop_size=None, root=None, fold=None):
        self.config = config
        self.df = pd.read_csv(csv_path)
        self.base_dset = self.df[self.df['fold'] == fold]
        self.root = root
        
        self.preprocessor = get_preprocessor(size=size,
                                            crop_size=crop_size,
                                            random_crop=random_crop,
                                            additional_targets={"previous": "image"})
        self.crop_size = crop_size
        if self.crop_size is not None:
            self.rescaler = albumentations.Compose(
                [albumentations.SmallestMaxSize(max_size = self.crop_size)],
                additional_targets={"previous": "image"})

    def __len__(self):
        return len(self.base_dset)

    def __getitem__(self, i):
        e = defaultdict()
        item = self.base_dset[i]
        assert len(item) == 1
        cur_path = item['current path'].tolist()[0]
        pre_path = item['previous path'].tolist()[0]
        e['image'] = Image.open(os.path.join(self.root, cur_path)).convert('RGB')
        e['previous'] = Image.open(os.path.join(self.root, pre_path)).convert('RGB')
        transformed = self.preprocessor(image=e['image'], previous=e['previous'])
        e['image'] = transformed['image']
        e['previous'] = transformed['previous']
        e['interval'] = item['interval'].tolist()[0]
        return e


class TrainWithPrevious(BaseWithPrevious):
    # default to random_crop=True
    def __init__(self, random_crop=True, fold='train', **kwargs):
        super().__init__(random_crop=random_crop, fold=fold, **kwargs)


class ValidationWithPrevious(BaseWithPrevious):
    def __init__(self, fold='valid', **kwargs):
        super().__init__(fold=fold, **kwargs)
