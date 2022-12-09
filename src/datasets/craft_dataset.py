from PIL import ImageFile
import torch
import random
import pickle
import cv2
import json
import zlib
import numpy as np
from time import time
from PIL import Image
from pathlib import Path
from itertools import chain
from torchvision import transforms

from datasets.make_charbox import PseudoCharBoxBuilder
from utils.craft_word2char_utils import (crop_image, divide_region, watershed,
                                         find_box, cal_confidence, reorder_points,
                                         cal_affinity_boxes, img_normalize,
                                         load_image, GaussianGenerator)
from utils.craft_utils import resize_aspect_ratio, normalizeMeanVariance
from utils.data_manipulation import (generate_affinity, generate_target)


# 다양한 형태의 한글 문자 OCR
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(42)

class AIhub_Dataset(torch.utils.data.Dataset):
    def __init__(self, original_data_dir=None, data_list_file=None, data_list=None):
        super(AIhub_Dataset, self).__init__()
        self.data_dir = Path(original_data_dir)
        if data_list_file is not None:
            with open(data_list_file,'r') as f:
                self.json_names = f.readlines()

        if data_list is not None:
            self.json_names = data_list
        #self.data_list = [None] * len(self.json_names)

    def __len__(self):
        return len(self.json_names)

    def __getitem__(self, idx):
        json_fn = self.data_dir / ('2.라벨링데이터/OCR(public)')/ (self.json_names[idx].strip()+'.json')
        image, word_boxes, words, image_fn = self.load_data(json_fn, self.json_names[idx].strip())

        return image, word_boxes, words, image_fn

    def load_data(self, json_fn: Path, file_fn: str):
        """
        return:
            PIL.image, wordBoxes
        """
        with json_fn.open('r', encoding='utf8') as f:
            data = json.load(f)
        image_fn = self.data_dir / ('1.원천데이터/OCR(public)') / (file_fn +'.jpg')

        if not image_fn.exists():
            print("FileNotFoundError: image_fn : {}".format(image_fn))
            raise FileNotFoundError
        image = load_image(image_fn)

        word_boxes, words = [], []
        for bbox in data["Bbox"]:
            text = bbox["data"]
            x,y =  bbox["x"], bbox["y"]
            lx, ly, rx, ry = min(x), min(y), max(x), max(y)
            if lx >= rx or ly >= ry:
                continue
            word_boxes.append([[lx, ly], [rx, ly], [rx, ry], [lx, ry]])
            words.append(text)

        return image, np.array(word_boxes), words, image_fn

# 다양한 형태의 한글 문자 OCR

class AIhub_collate(object):
    def __init__(self, cfg=None):
        self.image_size = cfg.image_size
        self.cfg = cfg

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def __call__(self, batch):
        """
        preprocess batch
        """
        batch_big_image, batch_weight_character, batch_weight_affinity = [], [], []
        for image, word_boxes, words, fn in batch:
            prepared_path = fn.parent / f"{fn.stem}_{self.image_size}_prepared.pkl"
            if prepared_path.exists():
                with prepared_path.open('rb') as f:
                    temp = pickle.load(f)
                big_image = np.frombuffer(zlib.decompress(temp['big_image']),dtype=np.uint8)
                weight_character = np.frombuffer(zlib.decompress(temp['weight_character']),dtype=np.float32)
                weight_affinity = np.frombuffer(zlib.decompress(temp['weight_affinity']),dtype=np.float32)
                big_image = big_image.reshape(self.cfg.image_size, self.cfg.image_size, 3)
                weight_character = weight_character.reshape(self.cfg.image_size // 2, self.cfg.image_size//2)
                weight_affinity = weight_affinity.reshape(self.cfg.image_size // 2, self.cfg.image_size//2)
                batch_big_image.append(self.image_transform(Image.fromarray(big_image)))
                batch_weight_character.append(weight_character)
                batch_weight_affinity.append(weight_affinity)

            else:
                assert "Please Doing Preprocess First(python preprocess.py)"

        return  torch.from_numpy(np.stack(batch_big_image)),  \
                torch.from_numpy(np.stack(batch_weight_character)),  \
                torch.from_numpy(np.stack(batch_weight_affinity))

class AIhub_collate_preprocess(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, batch):
        """
        preprocess batch
        """
        return batch