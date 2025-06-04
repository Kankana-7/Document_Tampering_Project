import torch
from albumentations import ToTensorV2
import torchvision
import cv2
import jpegio
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import io
from io import BytesIO
from tempfile import NamedTemporaryFile
from PIL import ImageOps
import streamlit as st
import asyncio
import streamlit as st
from PIL import Image
from io import BytesIO

def crop_img(img, jpg_dct, crop_size=512, mask=None):
    if mask is None:
        use_mask = False
    else:
        use_mask = True
        crop_masks = []

    h, w, c = img.shape
    h_grids = h // crop_size
    w_grids = w // crop_size

    crop_imgs = []
    crop_jpe_dcts = []

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_img = img[y1:y2, x1:x2, :]
            crop_imgs.append(crop_img)
            crop_jpe_dct = jpg_dct[y1:y2, x1:x2]
            crop_jpe_dcts.append(crop_jpe_dct)
            if use_mask:
                if mask[y1:y2, x1:x2].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w % crop_size != 0:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_imgs.append(img[y1:y2, w - 512 : w, :])
            crop_jpe_dcts.append(jpg_dct[y1:y2, w - 512 : w])
            if use_mask:
                if mask[y1:y2, w - 512 : w].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if h % crop_size != 0:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            crop_imgs.append(img[h - 512 : h, x1:x2, :])
            crop_jpe_dcts.append(jpg_dct[h - 512 : h, x1:x2])
            if use_mask:
                if mask[h - 512 : h, x1:x2].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w % crop_size != 0 and h % crop_size != 0:
        crop_imgs.append(img[h - 512 : h, w - 512 : w, :])
        crop_jpe_dcts.append(jpg_dct[h - 512 : h, w - 512 : w])
        if use_mask:
            if mask[h - 512 : h, w - 512 : w].max() != 0:
                crop_masks.append(1)
            else:
                crop_masks.append(0)

    if use_mask:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, crop_masks
    else:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, None

def combine_img(imgs, h_grids, w_grids, img_h, img_w, crop_size=512):
    i = 0
    re_img = np.zeros((img_h, img_w))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, x1:x2] = imgs[i]
            i += 1

    if w_grids * crop_size < img_w:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, img_w - 512 : img_w] = imgs[i]
            i += 1

    if h_grids * crop_size < img_h:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            re_img[img_h - 512 : img_h, x1:x2] = imgs[i]
            i += 1

    if w_grids * crop_size < img_w and h_grids * crop_size < img_h:
        re_img[img_h - 512 : img_h, img_w - 512 : img_w] = imgs[i]

    return re_img

def slice_image_to_patches(orig_img: Image.Image, crop_size: int = 512, zoom_factor: float = 1.1, overlap: float = 0.7):
    """
    Slices a zoomed image into overlapping patches.

    Args:
        orig_img (PIL.Image): Original image (RGB).
        crop_size (int): Size of square patch (e.g., 512).
        zoom_factor (float): Zoom factor to apply before slicing (default 1.1).
        overlap (float): Overlap ratio between patches (default 0.7 means 70%).

    Returns:
        dict: Dictionary of cropped image patches as BytesIO objects.
              Keys are in format 'Cropped Image row_col.jpg'.
    """

    # Step 1: Zoom the image
    zoomed_width = int(orig_img.width * zoom_factor)
    zoomed_height = int(orig_img.height * zoom_factor)
    zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

    # Step 2: Determine slicing parameters
    stride = int(crop_size * overlap)
    width, height = zoomed_img.size

    y_positions = list(range(0, height - crop_size + 1, stride))
    if (height - crop_size) % stride != 0:
        y_positions.append(height - crop_size)

    x_positions = list(range(0, width - crop_size + 1, stride))
    if (width - crop_size) % stride != 0:
        x_positions.append(width - crop_size)

    # Step 3: Crop patches
    cropped_images = {}
    for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
        box = (x, y, x + crop_size, y + crop_size)
        cropped_patch = zoomed_img.crop(box)

        img_io = BytesIO()
        cropped_patch.save(img_io, format="JPEG")
        img_io.seek(0)

        row = (idx // len(x_positions)) + 1
        col = (idx % len(x_positions)) + 1
        key = f"Cropped Image {row}_{col}.jpg"
        # key = f"{row}_{col}"
        cropped_images[key] = img_io

    return cropped_images
