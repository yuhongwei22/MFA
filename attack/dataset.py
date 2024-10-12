import glob
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
# import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

def make_batch(image, mask):
    image_path = image
    image = np.array(Image.open(image_path).convert("RGB"))
    image = image.astype(np.float32)/255.0
    # image = image[None].transpose(0,3,1,2)
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image, "image_path":image_path}
    # for k in batch:
    #     batch[k] = batch[k].to(device=device)
    #     batch[k] = batch[k]*2.0-1.0
    return batch
class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.png', pad_out_to_modulo=None, scale_factor=None, data_len=-1):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.data_len = data_len
        self.scale_factor = scale_factor

    def __len__(self):
        if self.data_len == -1:
            return len(self.mask_filenames)
        else:
            return min(self.data_len, len( self.mask_filenames))

    def __getitem__(self, i):
        # image = load_image(self.img_filenames[i], mode='RGB')
        # mask = load_image(self.mask_filenames[i], mode='L')
        
        # result = dict(image=image, mask=mask[None, ...])

        # if self.scale_factor is not None:
        #     result['image'] = scale_image(result['image'], self.scale_factor)
        #     result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        # if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
        #     result['unpad_to_size'] = result['image'].shape[1:]
        #     result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
        #     result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        batch = make_batch(self.img_filenames[i],self.mask_filenames[i])

        return batch

class PrecomputedInpaintingResultsDataset(InpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result
    