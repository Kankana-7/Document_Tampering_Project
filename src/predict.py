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
from src.model_load import load_segmentation_model  
from src.image_processing import crop_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, new_qtb, totsr, toctsr = load_segmentation_model(device=device)

def predict_mask(cropped_images: dict, model, device: torch.device, crop_size: int = 512) -> dict:
    """
    Predict tampering masks for each cropped image.

    Args:
        cropped_images (dict): Keys are image names; values are BytesIO objects.
        model: The tampering detection model (must have .eval() mode).
        device (torch.device): 'cuda' or 'cpu'.
        crop_size (int): Size of the crop (default 512).

    Returns:
        dict: Keys are image names; values are predicted mask arrays (np.ndarray, shape [H, W, 3]).
    """
    model.eval()
    predicted_masks = {}

    for img_key, img_io in cropped_images.items():
        current_img = Image.open(img_io).convert("RGB")
        imgs_ori = np.array(current_img)

        # Save image temporarily for JPEG I/O
        with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

        # JPEG DCT extraction
        jpg_dct = jpegio.read(temp_path)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        h, w, c = imgs_ori.shape
        imgs_d = imgs_ori[0:(h // 8) * 8, 0:(w // 8) * 8, :].copy()
        dct_d = dct_ori[0:(h // 8) * 8, 0:(w // 8) * 8].copy()

        qs = torch.LongTensor(use_qtb2)
        crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=crop_size, mask=None)

        img_list = []
        for idx, crop in enumerate(crop_imgs):
            crop_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            data = toctsr(crop_rgb)
            dct = torch.LongTensor(crop_jpe_dcts[idx])

            data, dct, qs_batch = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
            dct = torch.abs(dct).clamp(0, 20)
            qs_batch = qs_batch.reshape(1, 1, 8, 8)

            with torch.no_grad():
                if data.shape[-2:] == (crop_size, crop_size):
                    pred = model(data, dct, qs_batch)
                    pred = torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu()
                    mask = (pred.numpy() * 255).astype(np.uint8)
                    img_list.append(mask)

        if img_list:
            ci = img_list[0].squeeze()
            if len(ci.shape) == 2:
                ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel RGB-style

            predicted_masks[img_key] = ci

    return predicted_masks

def process_displayed_masks_and_extract_text(predicted_masks, cropped_images,
                                             crop_size=512, overlap_crop_size=0.29,
                                             bottom_crop_size=0.29, extra_crop_5th_col=155):
    """
    Process the cropped & displayed predicted masks, compute bounding boxes,
    and extract text from corresponding regions in original cropped images.

    Returns:
        - bbox_extracted_texts: dict with {key: {(x, y, w, h): text}}
        - bbox_mask_with_text: dict with mask+box overlays
        - bbox_original_with_text: dict with original+box overlays
    """


    base_left_crop = int(crop_size * overlap_crop_size)
    bottom_crop = int(crop_size * bottom_crop_size)
    top_crop_row7 = int(crop_size * 0.21)
    min_area = 500

    bbox_extracted_texts = {}
    bbox_mask_with_text = {}
    bbox_original_with_text = {}

    for key in predicted_masks:
        row, col = map(int, key.split()[2].replace(".jpg", "").split("_"))
        
        # Load and crop predicted mask
        mask = predicted_masks[key]
        mask_img = Image.fromarray(mask).convert("RGB")
        orig_img = Image.open(cropped_images[key]).convert("RGB")

        # Horizontal crop
        if col > 1:
            left_crop = base_left_crop + extra_crop_5th_col if col == 5 else base_left_crop
            mask_img = mask_img.crop((left_crop, 0, crop_size, crop_size))
            orig_img = orig_img.crop((left_crop, 0, crop_size, crop_size))

        # Vertical crop
        if row == 7:
            mask_img = mask_img.crop((0, top_crop_row7, mask_img.width, crop_size))
            orig_img = orig_img.crop((0, top_crop_row7, orig_img.width, crop_size))
        else:
            mask_img = mask_img.crop((0, 0, mask_img.width, crop_size - bottom_crop))
            orig_img = orig_img.crop((0, 0, orig_img.width, crop_size - bottom_crop))

        # Convert to numpy for OpenCV
        mask_np = np.array(mask_img)
        original_np = np.array(orig_img)

        # Process the mask to get binary map
        mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)

        # Clean mask
        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_with_boxes = mask_np.copy()
        original_with_boxes = original_np.copy()
        extracted_texts = {}

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(cv2.convexHull(contour))

                # Draw rectangles
                cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(original_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract text from the corresponding region in original
                roi = original_np[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(roi_bin, config="--psm 6").strip()

                # Save text with coordinates
                extracted_texts[(x, y, w, h)] = text

                # Annotate text
                cv2.putText(mask_with_boxes, text, (x, y - 5 if y > 10 else y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Store outputs
        bbox_extracted_texts[key] = extracted_texts
        bbox_mask_with_text[key] = mask_with_boxes
        bbox_original_with_text[key] = original_with_boxes

    return bbox_extracted_texts, bbox_mask_with_text, bbox_original_with_text
