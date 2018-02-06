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
VesselProvider은 VesselDNN의 Input값에 해당하는 patch들을 만들어주는 클래스이다.

순서는 크게 4가지 단계로

1. 이미지를 읽어옴
2. 이미지 전역에 CLAHE 알고리즘으로 이미지 전처리함
3. 이미지 Augmenting 함
   - rescale / rotate / flip / gamma_correct 을 적용함
4. 이미지를 27x27 의 patch로 분할함
   - 안구 이미지가 포함되지 않은 부분이 들어가지 않도록, mask 영역 내에 포함된 것들만 가져옴
5. 전체 patch 중에서 n 만큼 Sampling함
   - Vessel인 Patch가 Non-Vessel인 Patch보다 훨씬 적으므로
     Vessel : non-Vessel = 0.3(label_ratio) : 0.7(1-label_ratio) 이 되도로 Sampling 함
6. Patch 별로 Global Contrast Normalization(GCN) 알고리즘을 적용함

argument :
    - data_dir : data가 있는 directory으로, 아래와 같은 폴더 구성을 따라야 함
        data_dir/
            -- label : 혈관 라벨링이 된 이미지 폴더
            -- mask : 안저 부분이 마스킹된 이미지 폴더
            -- image : 안저 이미지 폴더
    - kwargs :
        -- is_training : is training이 True이면, Data Augumenting을 하고, 아니면 하지 않음
        -- patch_size : patch의 크기 결정
        -- label_ratio : 혈관 patch의 비율

Usage:
    vesselprovider = VesselProvider("./data/")
    training_set = vesselprovider(200000)
'''
class VesselProvider(object):
    file_list = []
    max_index = 0
    index = 0
    is_training = False
    patch_size = 27
    label_ratio = 0.3

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.is_training = kwargs.get('is_training', False)
        self.patch_size = kwargs.get('patch_size', 27)
        self.label_ratio = kwargs.get('label_ratio', 0.3)
        self.clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

        self.label_dir = os.path.join(self.data_dir, 'label/')
        self.mask_dir = os.path.join(self.data_dir, 'mask/')
        self.img_dir = os.path.join(self.data_dir, 'image/')
        self.file_list = [path for path in os.listdir(
            self.img_dir) if os.path.splitext(path)[1] == '.png']

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
        # 이미지 내에서 Patch에 포함되지 않아야 하는 부분을 지정하는 mask
        # 예를 들어, 영상 내 바깥 검은색 부분은 학습에 유효하지 않으므로 빼주는 것이 좋음.
        img_path = os.path.join(self.mask_dir, file_name)
        raw = cv2.imread(img_path, 0)
        dst = np.zeros_like(raw)
        cv2.normalize(raw, dst, 0, 255, cv2.NORM_MINMAX)

        # Patch 단위로 쪼갤 때, 가장자리 쪽 이미지들을 빼주기 위함
        pad_size = int((self.patch_size // 2 + 1) * 1.5)
        blank = np.ones_like(dst)
        blank[:pad_size, ...] = 0
        blank[-pad_size:, ...] = 0
        blank[:, :pad_size, ...] = 0
        blank[:, -pad_size:, ...] = 0
        dst = cv2.bitwise_and(dst, dst, mask=blank)
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

        # Augmenting 과정에서 라벨 값들에 변동이 생김. 이것을 255으로 다시 잡아주는 과정
        label = (label > 1e-2) * 255
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
        pad_size = math.ceil(self.patch_size / 2)

        vessel_batch_size = int(batch_size * self.label_ratio)
        vessel_patches = []
        vessel_arg_list = random.choices(
            np.argwhere(label & mask), k=vessel_batch_size)
        for x, y in vessel_arg_list:
            patch = img[x - pad_size + 1:x + pad_size,
                        y - pad_size + 1:y + pad_size]
            vessel_patches.append(patch)

        neg_batch_size = batch_size - vessel_batch_size
        neg_patches = []
        neg_batch_size = random.choices(
            np.argwhere((~label) & mask), k=neg_batch_size)
        for x, y in neg_batch_size:
            patch = img[x - pad_size + 1:x + pad_size,
                        y - pad_size + 1:y + pad_size]
            neg_patches.append(patch)
        return vessel_patches, neg_patches

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
        std_hsv = (hsv[..., 2] - hsv[..., 2].mean()) / hsv[..., 2].std()
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
                (norm_img.max() - norm_img.min())
        return norm_img

    def __call__(self, n):
        # n개 만큼의 Patch를 생성
        imgs = []
        masks = []
        batch_size = n // self.max_index  # 이미지 별로 가져오는 Patch의 갯수
        patches_list = []
        label_dataset = []
        for i in range(self.max_index):
            label, mask, img = self._get_next_image()  # 이미지를 불러옴
            img = self._apply_clahe(img)  # 이미지에 CLAHE 알고리즘 적용
            if self.is_training:
                label, mask, img = self._augument_data(
                    label, mask, img)  # training인 경우, Augumenting를 적용
            if self.max_index == i - 1:
                # 나눠 떨어지지 않은 경우 나눠 떨어지지 않은 경우 마지막 이미지에서 나머지 만큼을 더 가져옴
                remain_batch = n - batch_size * self.max_index
                batch_size += remain_batch

            vessel_patches, neg_patches = self._extract_patches_in_image(
                label, mask, img, batch_size)  # 이미지에서 batch_size 만큼 쪼개서 가져옴

            # one-hot으로 labeling list 생성
            vessel_size, neg_size = len(vessel_patches), len(neg_patches)
            vessel_labels, neg_labels = [[0, 1]] * \
                vessel_size, [[1, 0]] * neg_size

            # Random Shuffling
            patches = vessel_patches + neg_patches
            labels = vessel_labels + neg_labels
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

        # convert from list to numpy array
        patch_dataset = np.stack(patch_dataset)
        label_dataset = np.array(label_dataset)
        return patch_dataset, label_dataset


class vesselDraw(object):
    def __init__(self, meta_graph_path, variable_path):
        self.batch_size = 256

        # Tensorflow Model Restoring
        tf.reset_default_graph()
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_graph_path)
        init = [tf.global_variables_initializer(
        ), tf.local_variables_initializer()]
        self.sess.run(init)
        saver.restore(self.sess, variable_path)
        self.graph = tf.get_default_graph()
        self.softmax = self.graph.get_tensor_by_name("softmax:0")

    def run_vessel_segmentation(self, img):
        patches = self.get_input_patches(img)
        y_list = []
        for idx in range(len(patches) // self.batch_size + 1):
            x = np.stack(patches[self.batch_size *
                                 idx:self.batch_size * (idx + 1)])
            y = self.sess.run(self.softmax, feed_dict={
                              "input:0": x, "keep_prob:0": 1.})
            y_list.append(y)
        result = np.concatenate(y_list)

        output_size = int(np.sqrt(len(result)))

        vessel = result[..., 1].reshape((output_size, output_size))
        vessel = np.pad(vessel, 14, mode='constant')[:-1, :-1]
        vessel = np.stack([1 - vessel, vessel], axis=-1)
        return vessel

    def get_input_patches(self, img):
        img = self.apply_clahe(img)
        patches = self.extract_patches_in_image(img)
        patches = [self.apply_gcn(patch) for patch in patches]
        return patches

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def extract_patches_in_image(self, img, patch_size=27):
        pad_size = math.ceil(patch_size / 2)
        patches = []
        for x in range(pad_size, img.shape[0] - pad_size + 1):
            for y in range(pad_size, img.shape[1] - pad_size + 1):
                patch = img[x - pad_size:x + pad_size -
                            1, y - pad_size:y + pad_size - 1]
                patches.append(patch)
        return patches

    def normalize_data(self, img):
        norm_img = np.zeros_like(img)
        try:
            norm_img = cv2.normalize(img, norm_img, alpha=0, beta=1,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        except TypeError as e:
            norm_img = (norm_img - norm_img.min()) / \
                (norm_img.max() - norm_img.min())
        return norm_img

    def apply_gcn(self, patch):
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        blank = np.zeros_like(hsv[..., 2])
        std_hsv = (hsv[..., 2] - hsv[..., 2].mean()) / hsv[..., 2].std()
        hsv[..., 2] = cv2.normalize(std_hsv, blank, 0, 255, cv2.NORM_MINMAX)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # apply gaussian blur to reduce noise
        result = cv2.GaussianBlur(result, (3, 3), 0)
        result = self.normalize_data(result)
        return result

    def apply_crf(self, softmax, image, **kwargs):
        transposed_softmax = softmax.transpose((2, 0, 1))
        height, width = image.shape[:2]

        nlabels = kwargs.get("nlabels", 2)
        infer_nums = kwargs.get("infer_nums", 3)

        sxy = kwargs.get("sxy", (80, 80))
        srgb = kwargs.get("srgb", (13, 13, 13))
        compat = kwargs.get("compat", 2)

        kernel = kwargs.get("kernel", dcrf.DIAG_KERNEL)
        normalization = kwargs.get("normalization", dcrf.NORMALIZE_SYMMETRIC)

        # DenseCRF 선언
        dense_crf = dcrf.DenseCRF2D(width, height, nlabels)
        # Unary Potential Setting
        unary = unary_from_softmax(transposed_softmax)
        dense_crf.setUnaryEnergy(unary)
        # Pairwise Potential Setting
        dense_crf.addPairwiseBilateral(sxy=sxy,
                                       srgb=srgb,
                                       rgbim=image,
                                       compat=compat,
                                       kernel=kernel,
                                       normalization=normalization)
        # Inferencing
        Q = dense_crf.inference(infer_nums)
        res = np.argmax(Q, axis=0).reshape((height, width))
        return res

    def __call__(self, img_path):
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
        elif isinstance(img_path, np.ndarray):
            img = img_path
            img = cv2.resize(img, (512, 512))

        vessel_softmax = self.run_vessel_segmentation(img)
        result = self.apply_crf(vessel_softmax, img)

        return result, vessel_softmax
