# Overview

Original CRAFT text detector's input image size is `384x384`. Though CRAFT showed good performance for scene text detection, the input size is not enough for the high-resolution task, especially when it comes to document.

This repository of CRAFT, you can change input image size for improving model performance at training.

This code is based on https://github.com/YongWookHa/craft-text-detector

## Difference from Original Code for speed up

1. cv2 is working on GPU ( You can install using attached Dockerfile ) 

2. preprocessed data format is changed from .npy to pkl using zlib.compress for saving storage

3. For linking to word bbox, use STRTree (from shapley)

4. In this subject, we had to make pseudo labels. So, preprocess code is changed.

5. You have to preprocess first, before start training

5. etc..

# How to use

## Prepare your data

First of all, write your own Dataloader code.

In `datasets/craft_dataset.py`, you can find `CustomDataset`.

Make your `CustomDataset` return `image, char_boxes, words, image_fn` by `__getitem__` method. Return data format should be same as below.

- `image` : np.ndarray  
- `char_boxes` : character level bounding box coord.
    ```
    [   [lx, ly], [rx, ly], [rx, ry], [lx, ry],
        [lx, ly], [rx, ly], [rx, ry], [lx, ry],
        ...]   
    ```  
- `words` : list of words. character annotation should be in order of bounding boxes.
- `image_fn` : [pathlib](https://docs.python.org/3/library/pathlib.html) image path  

Then, change setting in `settings/default.yaml`.

```
train_data_path: <your train data path>
val_data_path: <your validate data path>
```  

These two setting is all you need to edit.  

Now you are ready to train your model. But the training might be very slow because of data processing time at making character and affinity heatmap.   

When it comes to train detecting text in high resolution documents, the heatmap processing is very slow.

In fact, the same data processing repeats every epoch. So, it does not necessarily have to be done for every epoch. Therefore, let's preprocess it before we start training.

You can get pretrained model <a href=https://drive.google.com/file/d/1n3XD7rsXicK0ZSYyUdG4LHtnZrcBwdAZ/view?usp=sharing>here</a>

Run `preprocess.py`( MultiProcessing ) like below.

```bash
python preprocess.py --setting settings/default.yaml --num_workers 16 --batch_size 1 
```

## Train

```bash
python run.py --setting settings/default.yaml --version 0 --num_workers 16 -bs 4 --preprocessed
```

To monitor the training progress, use tensorboard.

```bash
tensorboard --logdir tb_logs --bind_all
```
