import cv2

import pandas as pd
import numpy as np

import os
import re
import glob
import random

from skimage.transform import rotate
from skimage.filters import gaussian

class DataProvider(object):
    file_list = []
    max_index = 0
    index = 0
    clahe = None
    def __init__(self, img_dir, *mask_dirs, **kwargs):
        self.img_dir = img_dir
        self.mask_dirs = mask_dirs
        self.rotate_range = kwargs.get('rotate_range', (1, 360))
        self.img_size = kwargs.get('img_size', (256, 256))
        self.g_size = kwargs.get('gaussian_size', 1)
        self.clipLimit = kwargs.get('clipLimit',2.0)
        self.tgsize = kwargs.get('tileGridSize',8)

        #For Pre-Processign (applying clahe)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=tuple([self.tgsize]*2))

        file_set = set(os.listdir(self.img_dir))
        for mask_dir in self.mask_dirs:
            file_set =  file_set & set(os.listdir(mask_dir))
        self.file_list = list(file_set)
        print('the number of input data : {}'.format(len(self.file_list)))
        self.max_index = len(self.file_list) - 1
        self.index = 0

    def _get_next_image(self):
        file_name = self.file_list[self.index]
        # if self.index is out of bound , shuffle list and initalize
        if self.index < self.max_index:
            self.index += 1
        else:
            random.shuffle(self.file_list)
            self.index = 0

        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self._get_masks(file_name)
        masks = self._append_inv_masks(masks)
        img, masks = self._preprocess_data(img, masks)
        return img, masks

    def _get_masks(self,file_name):
        mask_list = []
        for mask_dir in self.mask_dirs:
            mask_path = os.path.join(mask_dir, file_name)
            mask = cv2.imread(mask_path, 0)
            mask_list.append(mask)
        if len(mask_list) == 1:
            return np.expand_dims(mask,axis=-1)
        else:
            return np.stack(mask_list,axis=-1)

    def _append_inv_masks(self, masks):
        inv_masks = []
        for idx in range(masks.shape[-1]):
            inv_masks.append((~masks[...,idx]).copy())
        if len(inv_masks) == 1:
            inv_mask = inv_masks[0].copy()
        else:
            inv_mask = inv_masks[0].copy()
            for i in range(1,len(inv_masks)):
                inv_mask = cv2.bitwise_and(inv_mask,inv_masks[i])
        inv_mask = np.expand_dims(inv_mask,axis=-1)
        masks = np.concatenate([inv_mask,masks],axis=-1)
        return masks

    def _preprocess_data(self, img, masks):
        # clahe 적용
        img = self._apply_clahe(img)
        # resize to shape of output image
        img = self._resize_data(img)
        mask_list = []
        for idx in range(masks.shape[-1]):
            mask = self._resize_data(masks[...,idx])
            mask = self._normalize_data(mask)
            mask_list.append(mask)
        if len(mask_list) == 1:
            masks = np.expand_dims(mask,axis=-1)
        else:
            masks = np.stack(mask_list,axis=-1)
        return img, masks

    def _apply_clahe(self, img):
        # 명도를 기준으로 clahe 적용한 후, HSV -> RGB로 하여 이미지 복원
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        img2 = clahe.apply(hsv[...,2])
        hsv[...,2] = img2
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    def _normalize_data(self, img):
        norm_img = np.zeros_like(img)
        try:
            norm_img = cv2.normalize(img, norm_img, alpha=0, beta=1,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        except TypeError as e:
            norm_img = (norm_img - norm_img.min()) / \
                (norm_img.max()-norm_img.min())
        return norm_img

    def _resize_data(self, img):
        img = cv2.resize(img, self.img_size)
        return img

    def _augument_data(self, img, mask):
        # rotate image
        angle = random.randint(*(self.rotate_range))
        img = rotate(img, angle)
        for idx in range(mask.shape[-1]):
            cval = 1 if idx == 0 else 0
            mask[..., idx] = rotate(mask[..., idx], angle, cval=cval)
        # apply gaussian blur
        img = gaussian(img, sigma=self.g_size, multichannel=True)
        return img, mask

    def __call__(self, n):
        imgs = []
        masks = []
        for _ in range(n):
            img, mask = self._get_next_image()
            img, mask = self._augument_data(img, mask)
            imgs.append(img)
            masks.append(mask)

        batch_img = np.stack(imgs)
        batch_mask = np.stack(masks)
        return batch_img, batch_mask
