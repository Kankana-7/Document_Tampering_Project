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

# def inference_on_image(orig_img: Image.Image, model, device):
#     import time  # Ensure time module is available

#     print("Starting inference...", flush=True)
#     total_start_time = time.time()

#     orig_img = orig_img.convert("RGB")

#     # ------------------ Patch Generation ------------------
#     patch_start_time = time.time()

#     crop_size = 512
#     zoom_factor = 1.1
#     zoomed_width = int(orig_img.width * zoom_factor)
#     zoomed_height = int(orig_img.height * zoom_factor)
#     zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

#     stride = int(crop_size * 0.7)
#     width, height = zoomed_img.size
#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)
#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

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
#         cropped_images[key] = img_io

#     print(f"Patch generation: {time.time() - patch_start_time:.2f}s", flush=True)

#     # ------------------ Mask Prediction ------------------
#     mask_start_time = time.time()
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

#     print(f"Mask prediction: {time.time() - mask_start_time:.2f}s", flush=True)

#     # ------------------ Text Extraction ------------------
#     text_start_time = time.time()
#     bbox_texts, mask_boxes, original_boxes = process_displayed_masks_and_extract_text(
#         predicted_masks, cropped_images
#     )
#     print(f"Text extraction: {time.time() - text_start_time:.2f}s", flush=True)

#     # ------------------ Total Time ------------------
#     print(f"Total processing time: {time.time() - total_start_time:.2f}s", flush=True)

#     return bbox_texts

############################### A bit optimized ###########################

# def inference_on_image(orig_img: Image.Image, model, device):
#     import time  # Ensure time module is available

#     print("Starting inference...", flush=True)
#     total_start_time = time.time()

#     orig_img = orig_img.convert("RGB")

#     # ------------------ Patch Generation ------------------
#     patch_start_time = time.time()

#     crop_size = 512
#     zoom_factor = 1.1
#     zoomed_width = int(orig_img.width * zoom_factor)
#     zoomed_height = int(orig_img.height * zoom_factor)
#     zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

#     stride = int(crop_size * 0.7)
#     width, height = zoomed_img.size
#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)
#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

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
#         cropped_images[key] = img_io

#     print(f"Patch generation: {time.time() - patch_start_time:.2f}s", flush=True)

#     # ------------------ Mask Prediction ------------------
#     mask_start_time = time.time()
#     predicted_masks = {}

#     for img_key in cropped_images:
#         # 1) Load the 512×512 patch into a NumPy array
#         current_io = cropped_images[img_key]
#         current_img = Image.open(current_io)
#         imgs_ori = np.array(current_img)

#         # 2) Write it out once to a temp file so jpegio.read() can get DCT + Q-table
#         with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#             temp_path = temp_file.name
#             cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

#         jpg_dct = jpegio.read(temp_path)
#         os.remove(temp_path)  # clean up immediately

#         dct_ori = jpg_dct.coef_arrays[0].copy()
#         use_qtb2 = jpg_dct.quant_tables[0].copy()

#         # 3) Trim original to multiples of 8
#         h, w, _ = imgs_ori.shape
#         h8, w8 = (h // 8) * 8, (w // 8) * 8
#         imgs_d = imgs_ori[:h8, :w8].copy()
#         dct_d  = dct_ori[:h8, :w8].copy()

#         # 4) Run crop_img (it will return a list of sub-patches, but we only need the first)
#         qs = torch.LongTensor(use_qtb2)
#         crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(
#             imgs_d, dct_d, crop_size=512, mask=None
#         )

#         if not crop_imgs:
#             # No valid 512×512 sub-patch was returned
#             continue

#         # 5) Take only the very first sub-crop and its DCT
#         first_crop_rgb = Image.fromarray(cv2.cvtColor(crop_imgs[0], cv2.COLOR_BGR2RGB))
#         first_dct      = torch.LongTensor(crop_jpe_dcts[0])

#         # 6) Build tensors for a single forward pass
#         data_tensor = toctsr(first_crop_rgb).unsqueeze(0).to(device)     # (1×3×512×512)
#         dct_tensor  = first_dct.unsqueeze(0).to(device)                  # (1×512×512)
#         qs_tensor   = qs.unsqueeze(0).to(device)                          # (1×8×8)

#         dct_tensor = torch.abs(dct_tensor).clamp(0, 20)
#         B, C, H, W = data_tensor.shape
#         qs_tensor  = qs_tensor.reshape(B, 1, 8, 8)                        # (1×1×8×8)

#         # 7) Single forward + softmax
#         with torch.no_grad():
#             pred = model(data_tensor, dct_tensor, qs_tensor)
#             fg   = torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu()   # (1×512×512)

#         # 8) Convert to uint8 mask and make 3-channel
#         mask2d = (fg.numpy() * 255).astype(np.uint8)[0]                    # (512×512)
#         if mask2d.ndim == 2:
#             mask3c = np.stack([mask2d] * 3, axis=-1)
#         else:
#             mask3c = mask2d

#         predicted_masks[img_key] = mask3c

#     print(f"Mask prediction: {time.time() - mask_start_time:.2f}s", flush=True)

#     # ------------------ Text Extraction ------------------
#     text_start_time = time.time()
#     bbox_texts, mask_boxes, original_boxes = process_displayed_masks_and_extract_text(
#         predicted_masks, cropped_images
#     )
#     print(f"Text extraction: {time.time() - text_start_time:.2f}s", flush=True)

#     # ------------------ Total Time ------------------
#     print(f"Total processing time: {time.time() - total_start_time:.2f}s", flush=True)

#     return bbox_texts

#############################################################################
#############################################################################
def inference_on_image(orig_img: Image.Image, model, device):
    import time
    import jpegio
    from io import BytesIO
    from tempfile import NamedTemporaryFile
    import numpy as np
    import cv2
    from tqdm import tqdm
    import torch
    from concurrent.futures import ThreadPoolExecutor

    print("Starting inference...", flush=True)
    total_start_time = time.time()

    orig_img = orig_img.convert("RGB")

    # ------------------ Patch Generation ------------------
    patch_start_time = time.time()

    crop_size = 512
    zoom_factor = 1.1
    stride = int(crop_size * 0.7)
    zoomed_img = orig_img.resize(
        (int(orig_img.width * zoom_factor), int(orig_img.height * zoom_factor)),
        Image.LANCZOS
    )

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
        row = idx // len(x_positions) + 1
        col = idx % len(x_positions) + 1
        key = f"Cropped Image {row}_{col}.jpg"
        cropped_images[key] = img_io

    print(f"Patch generation: {time.time() - patch_start_time:.2f}s", flush=True)

    # ------------------ Mask Prediction ------------------
    mask_start_time = time.time()
    predicted_masks = {}

    def predict_mask_from_io(key_io_pair):
        key, img_io = key_io_pair
        current_img = Image.open(img_io).convert("RGB")
        img_np = np.array(current_img)

        # Save to temp file for jpegio
        with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        jpg_dct = jpegio.read(temp_path)
        os.remove(temp_path)

        dct_ori = jpg_dct.coef_arrays[0]
        q_table = jpg_dct.quant_tables[0]
        h, w, _ = img_np.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        img_np = img_np[:h8, :w8]
        dct_ori = dct_ori[:h8, :w8]

        qs = torch.LongTensor(q_table)
        crop_imgs, crop_jpe_dcts, _, _, _ = crop_img(
            img_np, dct_ori, crop_size=512, mask=None
        )

        if not crop_imgs:
            return key, None

        rgb_crop = crop_imgs[0]
        dct_crop = torch.LongTensor(crop_jpe_dcts[0])
        input_img = Image.fromarray(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))

        data_tensor = toctsr(input_img).unsqueeze(0).to(device)
        dct_tensor = dct_crop.unsqueeze(0).to(device).abs().clamp(0, 20)
        qs_tensor = qs.reshape(1, 1, 8, 8).to(device)

        with torch.no_grad():
            pred = model(data_tensor, dct_tensor, qs_tensor)
            fg = torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu()

        mask2d = (fg.numpy()[0] * 255).astype(np.uint8)
        mask3c = np.stack([mask2d] * 3, axis=-1)

        return key, mask3c

    with ThreadPoolExecutor(max_workers=8) as executor:
        for key, mask in tqdm(executor.map(predict_mask_from_io, cropped_images.items()), total=len(cropped_images)):
            if mask is not None:
                predicted_masks[key] = mask

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


img = Image.open("/home/xelpmoc/Documents/DocTamperAPI/1 - Name,Address,date,statecode_tempare.jpg")
all_texts = inference_on_image(img, model, device)
result = display_extracted_texts_with_explanations(all_texts)
print(result)
