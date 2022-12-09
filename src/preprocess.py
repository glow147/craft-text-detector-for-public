"""
CRAFT TEXT DETECTER

Custom CRAFT model trainer for high resolution document image using
pytorch-lightning.

author: YongWook Ha @ NHN Diquest
"""
import torch.multiprocessing as mp
import torch
import pickle
import cv2
import zlib
import numpy as np
import argparse
from models.craft import CRAFT
from datasets.craft_dataset import AIhub_Dataset, AIhub_collate_preprocess
from torch.utils.data import DataLoader
from utils.base import load_setting
from collections import OrderedDict
from datasets.make_charbox import PseudoCharBoxBuilder
from utils.data_manipulation import (generate_affinity, generate_target)
from tqdm import tqdm

@torch.no_grad()
def prepro(model, dataloader, cfg):
    for batches in tqdm(dataloader):
        for image, word_boxes, words, fn in batches:
            preprocess(cfg, model, image, word_boxes, words, fn)

def resize(image, character, big_side=768):

    height, width, channel = image.shape
    max_side = max(height, width)
    big_resize = (int(width/max_side*big_side), int(height/max_side*big_side))
    small_resize = (int(width/max_side*(big_side//2)), int(height/max_side*(big_side//2)))
    image = cv2.resize(image, big_resize)

    character = np.array(character)
    character[0, :, :] = character[0, :, :] * (small_resize[0] / width)
    character[1, :, :] = character[1, :, :] * (small_resize[1] / height)

    big_image = np.ones([big_side, big_side, 3], dtype=np.float32)*255
    h_pad, w_pad = (big_side-image.shape[0])//2, (big_side-image.shape[1])//2
    big_image[h_pad: h_pad + image.shape[0], w_pad: w_pad + image.shape[1]] = image
    big_image = big_image.astype(np.uint8)

    small_image = cv2.resize(big_image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    character[0, :, :] += (w_pad // 2)
    character[1, :, :] += (h_pad // 2)

    # character fit to small image
    return big_image, small_image, character

def preprocess(cfg, model, image, word_boxes, words, fn):
    prepared_path = fn.parent / f"{fn.stem}_{cfg.craft.image_size}_prepared.pkl"
    if not prepared_path.exists():
        pseudo = PseudoCharBoxBuilder(cfg.craft.WATERSHED)
        char_boxes, _, _, words_count = pseudo.build_char_box(model, 0, image, word_boxes, words, fn)

        char_boxes = np.transpose(char_boxes, (2, 1, 0))
        big_image, small_image, character = resize(image, char_boxes, big_side=1536)  # Resize the image

        # Generate character heatmap
        weight_character = generate_target(small_image.shape, character.copy())

        # Generate affinity heatmap
        weight_affinity, _ = generate_affinity(small_image.shape, character.copy(), words, words_count)

        weight_character = weight_character.astype(np.float32)
        weight_affinity = weight_affinity.astype(np.float32)

        outputs = {}
        outputs['big_image'] = zlib.compress(big_image)
        outputs['weight_character'] = zlib.compress(weight_character)
        outputs['weight_affinity'] = zlib.compress(weight_affinity)

        out_path = fn.parent / f"{fn.stem}_{cfg.craft.image_size}_prepared.pkl"
        with out_path.open('wb') as f:
            pickle.dump(outputs, f)

if __name__ == "__main__":
    num_process = mp.cpu_count() // 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="../settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=1,
                        help="Train experiment version")
    parser.add_argument("--num_workers", "-nw", type=int, default=4,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=1,
                        help="batch size")
    args = parser.parse_args()
    # setting
    cfg = load_setting(args.setting)
    device_num = 0
    cfg.craft.gpus = 1
    cfg.update(vars(args))
    print("setting:", cfg)

    model = CRAFT(cfg).to(torch.device(f'cuda:{device_num}'))
    saved = torch.load(cfg.craft.weight, map_location=torch.device(f'cuda:{device_num}'))
    model.load_state_dict(saved)
    model.eval()

    # 다양한 한글 데이터
    with open(cfg.train_data_list_file, 'r') as f:
        json_names = f.readlines()
    with open(cfg.valid_data_list_file, 'r') as f:
        json_names.extend(f.readlines())
    train_dataloader_batch = []
    total_len = len(json_names)
    step = int(total_len / num_process)
    custom_collate = AIhub_collate_preprocess(cfg)
    for n in range(num_process):
        split_data = json_names[step*n:step*(n+1)]
        train_set = AIhub_Dataset(original_data_dir = cfg.original_data_dir,
                                data_list_file = None,
                                data_list = split_data)
        train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers,
                                    collate_fn=custom_collate)
        train_dataloader_batch.append(train_dataloader)

    process = []
    mp.set_start_method('spawn')
    model.share_memory()

    for rank in range(num_process):
        p = mp.Process(target=prepro, args=(model, train_dataloader_batch[rank], cfg))
        p.start()


