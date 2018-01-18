#-*-coding utf-8-*-
import os
import re
import glob
import random

import cv2
import numpy as np

from skimage.transform import rotate
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma

class DataProvider(object):
    file_list = []
    max_index = 0
    index = 0
    clahe = None
    def __init__(self, img_dir, *mask_dirs, **kwargs):
        self.img_dir = img_dir
        self.mask_dirs = mask_dirs
        self.is_training = kwargs.get('is_training', True)
        self.rotate_range = kwargs.get('rotate_range', (-90, 90))
        self.gamma_range = kwargs.get('gamma_range',(0.25,4))
        self.lf_factor = kwargs.get('lf_factor',0.3)
        self.hf_factor = kwargs.get('hf_Factor',1.5)
        self.sigma_factor = kwargs.get('sigma', 10)
        
        self.img_size = kwargs.get('img_size', (256, 256))
        self.g_size = kwargs.get('gaussian_size', 1)
        self.clipLimit = kwargs.get('clipLimit',2.0)
        self.tgsize = kwargs.get('tileGridSize',8)

        #For Pre-Processign (applying clahe)
        self.clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=tuple([self.tgsize]*2))

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
        
        img = self._resize_data(img)
        mask_list = []
        for idx in range(masks.shape[-1]):
            mask = masks[...,idx]
            mask = self._resize_data(mask)
            mask_list.append(mask)
        if len(mask_list) == 1:
            masks = np.expand_dims(mask,axis=-1)
        else:
            masks = np.stack(mask_list,axis=-1)
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
        img = self._convert_float_to_uint(img)
        img = self._apply_clahe(img)
        img = self._apply_homomorphic(img)
        img = self._normalize_data(img)
        
        mask_list = []
        masks = (masks>0).astype(np.uint8)
        masks[...,0] = 1-masks[...,1:].sum(axis=-1)
        masks = np.clip(masks,0,1)
        for idx in range(masks.shape[-1]):
            mask = self._normalize_data(masks[...,idx])
            mask_list.append(mask)
        if len(mask_list) == 1:
            masks = np.expand_dims(mask,axis=-1)
        else:
            masks = np.stack(mask_list,axis=-1)
        return img, masks

    def _convert_float_to_uint(self, np_array):
        if np_array.dtype == np.float:
            np_array = (np_array * 255).astype(np.uint8)
            return np.clip(np_array, 0, 255)
        elif np_array.dtype == np.uint8:
            return np_array
        else:
            np_array = (np_array * 255).astype(np.uint8)
            return np.clip(np_array, 0, 255)
    
    def _apply_clahe(self, img):
        # 명도를 기준으로 clahe 적용한 후, HSV -> RGB로 하여 이미지 복원
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        img2 = self.clahe.apply(hsv[...,2])
        hsv[...,2] = img2
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    def _apply_homomorphic(self, img):
        img_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y = img_YUV[...,0]
        rows, cols = y.shape[:2]

        imgLog = np.log1p(np.array(y, dtype='float') / 255)
        M, N = 2*rows + 1, 2*cols + 1

        X, Y = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        Xc, Yc = np.ceil(N/2), np.ceil(M/2)
        gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

        LPF = np.exp(-gaussianNumerator / (2*self.sigma_factor**2))
        HPF = 1 - LPF

        LPF_shift = np.fft.ifftshift(LPF.copy())
        HPF_shift = np.fft.ifftshift(HPF.copy())

        img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
        img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
        img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))

        img_adjusting = self.lf_factor*img_LF[0:rows, 0:cols] + self.hf_factor*img_HF[0:rows, 0:cols]

        img_exp = np.expm1(img_adjusting)
        img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
        img_out = np.array(255*img_exp, dtype = 'uint8')

        img_YUV[:,:,0] = img_out
        return cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)
    
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
        img, mask = self._rotate(img, mask)
        # flip
        img, mask = self._flip(img, mask)
        # gamma correct
        img = self._gamma_correct(img)
        # apply gaussian blur
        img = gaussian(img, sigma=self.g_size, multichannel=True)
        return img, mask

    def _rotate(self, img, mask):
        rotate_factor = random.randint(*self.rotate_range)
        img = rotate(img, rotate_factor)
        for idx in range(mask.shape[-1]):
            cval = 1 if idx == 0 else 0
            mask[..., idx] = rotate(mask[..., idx], rotate_factor, cval=cval)
        return img, mask

    def _flip(self, img, mask):
        # Flipping Horizontally
        if bool(random.getrandbits(1)):
            img = img[:, ::-1, :]
            mask = mask[:, ::-1,:]
        # Flipping Vertically
        if bool(random.getrandbits(1)):
            img = img[::-1, ...]
            mask = mask[::-1, ...]
        return img, mask

    def _gamma_correct(self, img):
        # Gamma correction by raising pixels to a power in [0.25,4]
        gamma_factor = random.uniform(*self.gamma_range)
        img = adjust_gamma(img, gamma_factor)
        return  img

    def __call__(self, n):
        imgs = []
        masks = []
        original_imgs = []
        for _ in range(n):
            img, mask = self._get_next_image()
            if self.is_training:
                img, mask = self._augument_data(img, mask)
                original = img.copy()
            else:
                original = self._normalize_data(img)
            original_imgs.append(original)
            
            img, mask = self._preprocess_data(img, mask)
            imgs.append(img)
            masks.append(mask)
        batch_img = np.stack(imgs)
        batch_mask = np.stack(masks)
        batch_original = np.stack(original_imgs)
        return batch_img, batch_mask, batch_original