import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import io
from io import BytesIO
from tempfile import NamedTemporaryFile
from PIL import ImageOps
import streamlit as st
import asyncio
import streamlit as st
from PIL import Image
from io import BytesIO
from src.model_load import *
from src.image_processing import *
from src.predict import *
from src.reasons import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, new_qtb, totsr, toctsr = load_segmentation_model(device=device)

def inference_on_image(orig_img: Image.Image, model, device):
    import time  # Ensure time module is available

    print("Starting inference...", flush=True)
    total_start_time = time.time()

    orig_img = orig_img.convert("RGB")

    # ------------------ Patch Generation ------------------
    patch_start_time = time.time()

    crop_size = 512
    zoom_factor = 1.1
    zoomed_width = int(orig_img.width * zoom_factor)
    zoomed_height = int(orig_img.height * zoom_factor)
    zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

    stride = int(crop_size * 0.7)
    width, height = zoomed_img.size
    y_positions = list(range(0, height - crop_size + 1, stride))
    if (height - crop_size) % stride != 0:
        y_positions.append(height - crop_size)
    x_positions = list(range(0, width - crop_size + 1, stride))
    if (width - crop_size) % stride != 0:
        x_positions.append(width - crop_size)

    cropped_images = {}
    for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
        box = (x, y, x + crop_size, y + crop_size)
        cropped_image = zoomed_img.crop(box)
        img_io = BytesIO()
        cropped_image.save(img_io, format="JPEG")
        img_io.seek(0)
        main_img_num = (idx // len(x_positions)) + 1
        sub_img_num = (idx % len(x_positions)) + 1
        key = f"Cropped Image {main_img_num}_{sub_img_num}.jpg"
        cropped_images[key] = img_io

    print(f"Patch generation: {time.time() - patch_start_time:.2f}s", flush=True)

    # ------------------ Mask Prediction ------------------
    mask_start_time = time.time()
    predicted_masks = {}

    for img_key in cropped_images:
        current_img = Image.open(cropped_images[img_key])
        imgs_ori = np.array(current_img)

        with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

        jpg_dct = jpegio.read(temp_path)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        h, w, c = imgs_ori.shape
        imgs_d = imgs_ori[0:(h // 8) * 8, 0:(w // 8) * 8, :].copy()
        dct_d = dct_ori[0:(h // 8) * 8, 0:(w // 8) * 8].copy()

        qs = torch.LongTensor(use_qtb2)
        crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

        img_list = []
        for idx, crop in enumerate(crop_imgs):
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            data = toctsr(crop)
            dct = torch.LongTensor(crop_jpe_dcts[idx])
            data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
            dct = torch.abs(dct).clamp(0, 20)
            B, C, H, W = data.shape
            qs = qs.reshape(B, 1, 8, 8)

            with torch.no_grad():
                if data.size()[-2:] == torch.Size((512, 512)):
                    pred = model(data, dct, qs)
                    pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
                    img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

        if len(img_list) > 0:
            ci = img_list[0].squeeze()
            if len(ci.shape) == 2:
                ci = np.stack([ci] * 3, axis=-1)
            predicted_masks[img_key] = ci

    print(f"Mask prediction: {time.time() - mask_start_time:.2f}s", flush=True)

    # ------------------ Text Extraction ------------------
    text_start_time = time.time()
    bbox_texts, mask_boxes, original_boxes = process_displayed_masks_and_extract_text(
        predicted_masks, cropped_images
    )
    print(f"Text extraction: {time.time() - text_start_time:.2f}s", flush=True)

    # ------------------ Total Time ------------------
    print(f"Total processing time: {time.time() - total_start_time:.2f}s", flush=True)

    return bbox_texts

# def inference_on_image(orig_img: Image.Image, model, device):
#     # Ensure image is in RGB
#     orig_img = orig_img.convert("RGB")

#     # Zoom image
#     crop_size = 512
#     zoom_factor = 1.1
#     zoomed_width = int(orig_img.width * zoom_factor)
#     zoomed_height = int(orig_img.height * zoom_factor)
#     zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

#     # Crop settings
#     stride = int(crop_size * 0.7)
#     width, height = zoomed_img.size
#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)
#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

#     num_rows = len(y_positions)
#     gap = 50
#     bottom_crop = int(crop_size * 0.29)
#     row_height = crop_size - bottom_crop
#     grid_height = (num_rows * row_height) + ((num_rows - 1) * gap)

#     # Resize for display (optional)
#     max_width = 700
#     aspect_ratio = orig_img.width / orig_img.height
#     new_width = max_width
#     new_height = int(new_width / aspect_ratio)
#     resized_orig = orig_img.resize((new_width, new_height), Image.LANCZOS)

#     # Prepare Cropped Images
#     cropped_images = {}
#     for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
#         box = (x, y, x + crop_size, y + crop_size)
#         cropped_image = zoomed_img.crop(box)
#         img_io = BytesIO()
#         cropped_image.save(img_io, format="JPEG")
#         img_io.seek(0)
#         main_img_num = (idx // len(x_positions)) + 1
#         sub_img_num = (idx % len(x_positions)) + 1
#         key = f"Cropped Image {main_img_num}_{sub_img_num}.jpg"
#         # key = f"{main_img_num}_{sub_img_num}"
#         cropped_images[key] = img_io

#     predicted_masks = {}

#     for img_key in cropped_images:
#         current_img = Image.open(cropped_images[img_key])
#         imgs_ori = np.array(current_img)

#         with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#             temp_path = temp_file.name
#             cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

#         jpg_dct = jpegio.read(temp_path)
#         dct_ori = jpg_dct.coef_arrays[0].copy()
#         use_qtb2 = jpg_dct.quant_tables[0].copy()

#         h, w, c = imgs_ori.shape
#         imgs_d = imgs_ori[0:(h // 8) * 8, 0:(w // 8) * 8, :].copy()
#         dct_d = dct_ori[0:(h // 8) * 8, 0:(w // 8) * 8].copy()

#         qs = torch.LongTensor(use_qtb2)
#         crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

#         img_list = []
#         for idx, crop in enumerate(crop_imgs):
#             crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             data = toctsr(crop)
#             dct = torch.LongTensor(crop_jpe_dcts[idx])

#             data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
#             dct = torch.abs(dct).clamp(0, 20)
#             B, C, H, W = data.shape
#             qs = qs.reshape(B, 1, 8, 8)

#             with torch.no_grad():
#                 if data.size()[-2:] == torch.Size((512, 512)):
#                     pred = model(data, dct, qs)
#                     pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
#                     img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

#         if len(img_list) > 0:
#             ci = img_list[0].squeeze()
#             if len(ci.shape) == 2:
#                 ci = np.stack([ci] * 3, axis=-1)
#             predicted_masks[img_key] = ci

#     bbox_texts, mask_boxes, original_boxes = process_displayed_masks_and_extract_text(
#         predicted_masks, cropped_images
#     )

#     return bbox_texts

img = Image.open("/home/xelpmoc/Documents/DocTamperAPI/1 - Name,Address,date,statecode_tempare.jpg")
all_texts = inference_on_image(img, model, device)
result = display_extracted_texts_with_explanations(all_texts)
print(result)

