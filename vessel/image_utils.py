#-*-coding utf-8-*-
import glob
import os
import re
import math
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.transform import rotate, rescale
from skimage.exposure import adjust_gamma

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

'''
DataProvider


'''
class DataProvider(object):
    file_list = []
    max_index = 0
    index = 0
    is_training = False
    patch_size = 27
    us_ratio = 0.3

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.is_training = kwargs.get('is_training', False)
        self.patch_size = kwargs.get('patch_size', 27)
        self.label_ratio = kwargs.get('label_ratio',0.3)
        clipLimit = kwargs.get("clipLimit",2.)
        tileGridSize = kwargs.get("tileGridSize",8)
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize,tileGridSize))

        self.label_dir = os.path.join(self.data_dir, 'label/')
        self.mask_dir = os.path.join(self.data_dir, 'mask/')
        self.img_dir = os.path.join(self.data_dir, 'image/')
        self.file_list = [path for path in os.listdir(
            self.img_dir) if os.path.splitext(path)[1] == '.png']
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
        label = self._read_label(file_name)
        mask = self._read_mask(file_name)
        img = self._read_img(file_name)
        return label, mask, img

    def _read_label(self, file_name):
        img_path = os.path.join(self.label_dir, file_name)
        raw = cv2.imread(img_path, 0)
        dst = np.zeros_like(raw)
        cv2.normalize(raw, dst, 0, 255, cv2.NORM_MINMAX)
        return dst

    def _read_mask(self, file_name):
        img_path = os.path.join(self.mask_dir, file_name)
        raw = cv2.imread(img_path, 0)
        dst = np.zeros_like(raw)

        cv2.normalize(raw, dst, 0, 255, cv2.NORM_MINMAX)
        pad_size = int((self.patch_size//2+1)*1.5)

        blank = np.ones_like(dst)

        blank[:pad_size,...] = 0
        blank[-pad_size:,...] = 0
        blank[:,:pad_size,...] = 0
        blank[:,-pad_size:,...] = 0
        dst = cv2.bitwise_and(dst,dst,mask=blank)

        return dst

    def _read_img(self, file_name):
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _augument_data(self, label, mask, img):
        label, mask, img = self._rescale(label, mask, img)
        label, mask, img = self._rotate(label, mask, img)
        label, mask, img = self._flip(label, mask, img)
        img = self._gamma_correct(img)

        label = (label > 1e-2)*255  # label adjust
        return label, mask, img

    def _rescale(self, label, mask, img):
        # Scaling by a factor between 0.7 and 1.2
        rescale_factor = random.uniform(0.7, 1.2)
        label = rescale(label, rescale_factor, mode='reflect')
        mask = rescale(mask, rescale_factor, mode='reflect')
        img = rescale(img, rescale_factor, mode='reflect')
        return label, mask, img

    def _rotate(self, label, mask, img):
        # Rotating by an angle from [-90,90]
        rotate_factor = random.randint(-90, 90)
        label = rotate(label, rotate_factor)
        mask = rotate(mask, rotate_factor)
        img = rotate(img, rotate_factor)
        return label, mask, img

    def _flip(self, label, mask, img):
        # Flipping Horizontally
        if bool(random.getrandbits(1)):
            label = label[:, ::-1]
            mask = mask[:, ::-1]
            img = img[:, ::-1, :]
        # Flipping Vertically
        if bool(random.getrandbits(1)):
            label = label[::-1, ...]
            mask = mask[::-1, ...]
            img = img[::-1, ...]
        return label, mask, img

    def _gamma_correct(self, img):
        # Gamma correction by raising pixels to a power in [0.25,4]
        gamma_factor = random.uniform(0.25, 4)
        img = adjust_gamma(img, gamma_factor)
        return img

    def _extract_patches_in_image(self, label, mask, img, batch_size):
        pad_size = math.ceil(self.patch_size/2)

        pos_batch_size = int(batch_size * self.label_ratio)
        pos_patches = []
        pos_arg_list = random.choices(np.argwhere(label & mask),k=pos_batch_size)
        for x, y in pos_arg_list:
            patch = img[x-pad_size+1:x+pad_size, y-pad_size+1:y+pad_size]
            pos_patches.append(patch)

        neg_batch_size = batch_size - pos_batch_size
        neg_patches = []
        neg_batch_size = random.choices(np.argwhere((~label) & mask),k=neg_batch_size)
        for x, y in neg_batch_size:
            patch = img[x-pad_size+1:x+pad_size, y-pad_size+1:y+pad_size]
            neg_patches.append(patch)
        return pos_patches, neg_patches

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _apply_gcn(self, patch):
        # global contrast normalization per patch
        # convert RGB to HSV, apply GCN
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        blank = np.zeros_like(hsv[..., 2])
        std_hsv = (hsv[..., 2]-hsv[..., 2].mean())/hsv[..., 2].std()
        hsv[..., 2] = cv2.normalize(std_hsv, blank, 0, 255, cv2.NORM_MINMAX)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # apply gaussian blur to reduce noise
        result = cv2.GaussianBlur(result, (3, 3), 0)
        return result

    def _normalize_data(self, img):
        norm_img = np.zeros_like(img)
        try:
            norm_img = cv2.normalize(img, norm_img, alpha=0, beta=1,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        except TypeError as e:
            norm_img = (norm_img - norm_img.min()) / \
                (norm_img.max()-norm_img.min())
        return norm_img

    def __call__(self, n):
        imgs = []
        masks = []
        batch_size = n // self.max_index
        patches_list = []
        label_dataset = []
        for i in range(self.max_index):
            label, mask, img = self._get_next_image()
            img = self._apply_clahe(img)
            if self.is_training:
                label, mask, img = self._augument_data(label, mask, img)
            if self.max_index == i-1:
                remain_batch = n - batch_size*self.max_index
                batch_size += remain_batch
            pos_patches, neg_patches = self._extract_patches_in_image(label, mask, img, batch_size)
            # undersampling for label imbalancing
            pos_size = len(pos_patches)
            neg_size = len(neg_patches)
            pos_labels = [[0,1]]*pos_size
            neg_labels = [[1,0]]*neg_size

            patches = pos_patches + neg_patches
            labels = pos_labels + neg_labels
            dataset = list(zip(patches, labels))
            random.shuffle(dataset)

            patches, labels = list(zip(*dataset))
            patches_list.extend(list(patches))
            label_dataset.extend(labels)

        patch_dataset = []
        for patch in patches_list:
            patch = self._apply_gcn(patch)
            patch = self._normalize_data(patch)
            patch_dataset.append(patch)

        patch_dataset = np.stack(patch_dataset)
        label_dataset = np.array(label_dataset)
        return patch_dataset, label_dataset

def apply_crf(softmax, image, nlabels=2,infer_nums=10):
    transposed_softmax = softmax.transpose((2,0,1))
    height, width = image.shape[:2]

    # DenseCRF 선언
    dense_crf = dcrf.DenseCRF2D(width, height,nlabels)
    # Unary Potential Setting
    unary = unary_from_softmax(transposed_softmax)
    dense_crf.setUnaryEnergy(unary)
    # Pairwise Potential Setting
    dense_crf.addPairwiseBilateral(sxy=100,
                                   srgb=(1,1,1),
                                   rgbim=image,
                                   compat=1,
                                   kernel=dcrf.DIAG_KERNEL,
                                   normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Inferencing
    Q = dense_crf.inference(infer_nums)
    res = np.argmax(Q, axis=0).reshape((height, width))
    return res
