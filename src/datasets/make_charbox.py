"""
This code was implemented by https://github.com/gmuffiness
"""

import os
import random
import math

import numpy as np
import cv2
import torch
import pickle

from utils.craft_word2char_utils import img_normalize
from datasets.pseudo_label import exec_watershed_by_version
from collections import OrderedDict


class PseudoCharBoxBuilder:
    def __init__(self, watershed_param):
        self.watershed_param = watershed_param
        self.cnt = 0
        self.flag = False

    def crop_image_by_bbox(self, image, box, word):
        w = max(
            int(np.linalg.norm(box[0] - box[1])), int(np.linalg.norm(box[2] - box[3]))
        )
        h = max(
            int(np.linalg.norm(box[0] - box[3])), int(np.linalg.norm(box[1] - box[2]))
        )
        try:
            word_ratio = h / w
        except:
            import ipdb

            ipdb.set_trace()

        one_char_ratio = min(h, w) / (max(h, w) / len(word))

        # NOTE: criterion to split vertical word in here is set to work properly on IC15 dataset
        if word_ratio > 2 or (word_ratio > 1.6 and one_char_ratio > 2.4):
            # warping method of vertical word (classified by upper condition)
            horizontal_text_bool = False
            long_side = h
            short_side = w
            M = cv2.getPerspectiveTransform(
                np.float32(box),
                np.float32(
                    np.array(
                        [
                            [long_side, 0],
                            [long_side, short_side],
                            [0, short_side],
                            [0, 0],
                        ]
                    )
                ),
            )
            self.flag = True
        else:
            # warping method of horizontal word
            horizontal_text_bool = True
            long_side = w
            short_side = h
            M = cv2.getPerspectiveTransform(
                np.float32(box),
                np.float32(
                    np.array(
                        [
                            [0, 0],
                            [long_side, 0],
                            [long_side, short_side],
                            [0, short_side],
                        ]
                    )
                ),
            )
            self.flag = False

        warped = cv2.cuda.warpPerspective(image, M, (long_side, short_side))
        return warped.download(), M, horizontal_text_bool

    def inference_word_box(self, net, gpu, word_image):
        if net.training:
            net.eval()

        with torch.no_grad():
            word_img_torch = torch.from_numpy(
                img_normalize(word_image)
            )
            word_img_torch = word_img_torch.permute(2, 0, 1).unsqueeze(0)
            word_img_torch = word_img_torch.to(torch.device(f'cuda:{gpu}'))
            with torch.cuda.amp.autocast():
                word_img_scores, _ = net(word_img_torch)
        return word_img_scores

    def clip_into_boundary(self, box, bound):
        if len(box) == 0:
            return box
        else:
            box[:, :, 0] = np.clip(box[:, :, 0], 0, bound[1])
            box[:, :, 1] = np.clip(box[:, :, 1], 0, bound[0])
            return box

    def get_confidence(self, real_len, pseudo_len):
        if pseudo_len == 0:
            return 0.0
        return (real_len - min(real_len, abs(real_len - pseudo_len))) / real_len

    def split_word_equal_gap(self, word_img_w, word_img_h, word):
        width = word_img_w
        height = word_img_h

        width_per_char = width / len(word)
        bboxes = []
        for j, char in enumerate(word):
            if char == " ":
                continue
            left = j * width_per_char
            right = (j + 1) * width_per_char
            bbox = np.array([[left, 0], [right, 0], [right, height], [left, height]])
            bboxes.append(bbox)

        bboxes = np.array(bboxes, np.float32)
        return bboxes

    def cal_angle(self, v1):
        theta = np.arccos(min(1, v1[0] / (np.linalg.norm(v1) + 10e-8)))
        return 2 * math.pi - theta if v1[1] < 0 else theta

    def clockwise_sort(self, points):
        # returns 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
        v1, v2, v3, v4 = points
        center = (v1 + v2 + v3 + v4) / 4
        theta = np.array(
            [
                self.cal_angle(v1 - center),
                self.cal_angle(v2 - center),
                self.cal_angle(v3 - center),
                self.cal_angle(v4 - center),
            ]
        )
        index = np.argsort(theta)
        return np.array([v1, v2, v3, v4])[index, :]

    def build_char_box(self, net, gpu, image, word_bboxes, words, fn):
        words = [ word.strip().replace(" ","") for word in words ]

        word_preprocess = OrderedDict({ 'word_images' : [], 'word_img_sizes' : [], 'scales' : [], 'Ms' : [], 'horizontal_text_bools' : [] })
        word_images, Ms, horizontal_text_bools = [], [], []
        word_img_sizes, scales = [], []
        src = cv2.cuda_GpuMat()
        src.upload(image)
        for word_bbox, word in zip(word_bboxes, words):
            word_image, M, horizontal_text_bool = self.crop_image_by_bbox(src, word_bbox, word)
            # word_image, M, horizontal_text_bool = self.crop_image_by_bbox(image, word_bbox, word)
            scale = 128.0 / word_image.shape[0]
            word_image = cv2.resize(word_image, None, fx=scale, fy=scale)
            word_preprocess['word_images'].append(word_image)
            word_preprocess['word_img_sizes'].append(word_image.shape)
            word_preprocess['scales'].append(scale)
            word_preprocess['Ms'].append(M)
            word_preprocess['horizontal_text_bools'].append(horizontal_text_bool)

        word_images, word_img_sizes, scales, Ms, horizontal_text_bools = word_preprocess.values()

        results = {'pseudo_char_bbox':[], 'confidence':[], 'horizontal_text_bool':[], 'word_count':[]}

        for word, word_image, word_img_size, M, horizontal_text_bool, scale in \
                    zip(words, word_images, word_img_sizes, Ms, horizontal_text_bools, scales):
            scores = self.inference_word_box(net, gpu, word_image)
            word_img_h, word_img_w, _ = word_img_size
            
            region_score = scores[0, :, :, 0].cpu().numpy()
            region_score = np.uint8(np.clip(region_score, 0, 1) * 255)

            real_word_without_space = word
            real_char_len = len(real_word_without_space)

            region_score_rgb = cv2.resize(region_score, (word_img_w, word_img_h))
            region_score_rgb = cv2.cvtColor(region_score_rgb, cv2.COLOR_GRAY2RGB)

            pseudo_char_bbox = exec_watershed_by_version(
                self.watershed_param, region_score, word_image
            )

            # Used for visualize only
            # watershed_box = pseudo_char_bbox.copy()

            pseudo_char_bbox = self.clip_into_boundary(
                pseudo_char_bbox, region_score_rgb.shape
            )

            confidence = self.get_confidence(real_char_len, len(pseudo_char_bbox))

            if confidence <= 0.5:
                pseudo_char_bbox = self.split_word_equal_gap(word_img_w, word_img_h, word)
                confidence = 0.5

            if len(pseudo_char_bbox) != 0:
                index = np.argsort(pseudo_char_bbox[:, 0, 0])
                pseudo_char_bbox = pseudo_char_bbox[index]

            pseudo_char_bbox /= scale

            M_inv = np.linalg.pinv(M)
            for i in range(len(pseudo_char_bbox)):
                pseudo_char_bbox[i] = cv2.perspectiveTransform(
                    pseudo_char_bbox[i][None, :, :], M_inv
                )
            pseudo_char_bbox = self.clip_into_boundary(pseudo_char_bbox, image.shape)
            results['pseudo_char_bbox'].extend(pseudo_char_bbox)
            results['confidence'].append(confidence)
            results['horizontal_text_bool'].append(horizontal_text_bool)
            results['word_count'].append(pseudo_char_bbox.shape[0])

        return np.stack(results['pseudo_char_bbox']), results['confidence'], results['horizontal_text_bool'], results['word_count']