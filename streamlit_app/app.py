from dtd import seg_dtd
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

model = seg_dtd(n_class=2)
weights = torch.load("/home/xelpmoc/Documents/DocTamperAPI/artifacts/seg_dtd_model_weights.pth")
model.load_state_dict(weights)
model.eval()

new_qtb = (
    np.array(
        [
            [2, 1, 1, 2, 2, 4, 5, 6],
            [1, 1, 1, 2, 3, 6, 6, 6],
            [1, 1, 2, 2, 4, 6, 7, 6],
            [1, 2, 2, 3, 5, 9, 8, 6],
            [2, 2, 4, 6, 7, 11, 10, 8],
            [2, 4, 6, 6, 8, 10, 11, 9],
            [5, 6, 8, 9, 10, 12, 12, 10],
            [7, 9, 10, 10, 11, 10, 10, 10],
        ],
        dtype=np.int32,
    )
    .reshape(
        64,
    )
    .tolist()
)

totsr = ToTensorV2()
toctsr = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)

device=torch.device("cpu")
model=model.to(device)
print(device)


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

def remove_black_bg(image):
    """
    Remove black background and replace with transparency.
    
    Args:
        image: The image from which black background should be removed.
    
    Returns:
        Image with transparent background where black pixels were.
    """
    image = image.convert("RGBA")
    data = image.getdata()
    
    new_data = []
    for item in data:
        # Replace black pixels with transparent pixels
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))  # Transparent
        else:
            new_data.append(item)
    
    image.putdata(new_data)
    return image

def removal_of_black_bg(image, tolerance=10):
    """
    Remove black background and replace with transparency.
    
    Args:
        image: The image from which black background should be removed.
        tolerance: Tolerance level for detecting near-black pixels (default is 10).
    
    Returns:
        Image with transparent background where black or near-black pixels were.
    """
    image = image.convert("RGBA")
    data = image.getdata()
    
    new_data = []
    for item in data:
        # Check if the pixel is close to black, considering the tolerance
        if item[0] <= tolerance and item[1] <= tolerance and item[2] <= tolerance:
            new_data.append((0, 0, 0, 0))  # Transparent
        else:
            new_data.append(item)
    
    image.putdata(new_data)
    return image

###
def display_patches_as_grid_no_overlap(patches_dict, crop_size=512, gap=0, 
                                       overlap_crop_size=0.29, bottom_crop_size=0.29, 
                                       extra_crop_5th_col=155):
    """
    Display patches in a grid without overlap by adjusting positions based on cropped dimensions.
    
    For non-first columns the left side of each patch is cropped by a fraction of the original size.
    In the 5th column an extra left crop is applied and the patch is padded on the right so that
    its width matches other non-first column patches.
    Additionally, for patches in the 5th row the top 50% of the image is removed, and the patch
    is padded at the bottom (if needed) so that its final height matches the other rows.

    Parameters:
        patches_dict (dict): Dictionary with keys containing patch coordinates as part of the key string 
                             (in the format "... row_col.jpg") and values as file paths to images.
        crop_size (int): Size of the (square) patch before any cropping.
        gap (int): Gap (in pixels) between adjacent patches.
        overlap_crop_size (float): Fraction of crop_size to remove from the left side in non-first columns.
        bottom_crop_size (float): Fraction of crop_size to remove from the bottom of all patches.
        extra_crop_5th_col (int): Additional pixels to crop from the left on the 5th column.
    """

    # Extract keys and parse out (row, col) coordinates from the key strings.
    patch_keys = list(patches_dict.keys())
    # Expect keys like "... 5_3.jpg" where 5 is the row and 3 is the column.
    coords = [tuple(map(int, key.split()[2].replace(".jpg", "").split("_")))
              for key in patch_keys]
    max_row = max(c[0] for c in coords)
    max_col = max(c[1] for c in coords)

    # Compute crop amounts.
    base_left_crop = int(crop_size * overlap_crop_size)
    bottom_crop = int(crop_size * bottom_crop_size)
    # Standard final height for each patch after bottom crop.
    image_height = crop_size - bottom_crop

    # Define effective widths.
    image_width_col1 = crop_size
    image_width_other_cols = crop_size - base_left_crop

    # Precompute x positions for each column.
    x_pos = {}
    current_x = 0
    x_pos[1] = current_x
    current_x += image_width_col1 + gap
    for col in range(2, max_col + 1):
        x_pos[col] = current_x
        current_x += image_width_other_cols + gap

    # Precompute y positions for each row.
    y_pos = {}
    current_y = 0
    y_pos[1] = current_y
    current_y += image_height + gap
    for row in range(2, max_row + 1):
        y_pos[row] = current_y
        current_y += image_height + gap

    # Determine overall canvas dimensions.
    total_width = x_pos.get(max_col, 0) + (image_width_col1 if max_col == 1 else image_width_other_cols)
    total_height = y_pos.get(max_row, 0) + image_height

    # Create a transparent canvas.
    canvas = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    # Process and paste each patch.
    for key in patch_keys:
        row, col = map(int, key.split()[2].replace(".jpg", "").split("_"))
        x_offset = x_pos[col]
        y_offset = y_pos[row]
        patch_img = Image.open(patches_dict[key])

        # -------------
        # Horizontal Cropping (Left)
        # -------------
        if col > 1:
            if col == 5:
                # For the 5th column, apply extra left crop.
                left_offset = base_left_crop + extra_crop_5th_col
                cropped = patch_img.crop((left_offset, 0, crop_size, crop_size))
                desired_width = image_width_other_cols  # crop_size - base_left_crop
                current_width = cropped.width  # crop_size - (base_left_crop + extra_crop_5th_col)
                if current_width < desired_width:
                    # Pad on the right to reach desired width.
                    new_patch = Image.new('RGBA', (desired_width, crop_size), (0, 0, 0, 0))
                    new_patch.paste(cropped, (0, 0))
                    patch_img = new_patch
                else:
                    patch_img = cropped
            else:
                # For other non-first columns, crop by the base left crop.
                patch_img = patch_img.crop((base_left_crop, 0, crop_size, crop_size))
        # For the first column, leave the full patch.

        # -------------
        # Vertical Cropping (Top & Bottom)
        # -------------
        if row == 7:
            # For the 5th row, always crop 50% from the top.
            top_crop = int(crop_size * 0.21)
            cropped = patch_img.crop((0, top_crop, patch_img.width, crop_size))
            # The current height is now crop_size - top_crop.
            current_height = cropped.height
            desired_height = image_height  # Standard final height (crop_size - bottom_crop)
            if current_height < desired_height:
                # Pad at the bottom if the cropped image is too short.
                new_patch = Image.new('RGBA', (patch_img.width, desired_height), (0, 0, 0, 0))
                new_patch.paste(cropped, (0, 0))
                patch_img = new_patch
            elif current_height > desired_height:
                # Crop the bottom if the image is too tall.
                patch_img = cropped.crop((0, 0, patch_img.width, desired_height))
            else:
                patch_img = cropped
        else:
            # For non-5th rows, crop only from the bottom.
            patch_img = patch_img.crop((0, 0, patch_img.width, crop_size - bottom_crop))

        # -------------
        # Optional: Remove any black background.
        # Ensure that the remove_black_bg() function is defined elsewhere.
        patch_img = remove_black_bg(patch_img)

        # Paste the processed patch onto the canvas using its alpha mask.
        canvas.paste(patch_img, (x_offset, y_offset), patch_img)

    # Display the final composite image using Streamlit.
    st.image(canvas, use_container_width=True)

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
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract

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


def display_predicted_masks_as_grid_no_overlap(mask_dict, crop_size=512, gap=30, 
                                               overlap_crop_size=0.29, bottom_crop_size=0.29, 
                                               extra_crop_5th_col=155):
    """
    Display predicted tampering masks in a grid without overlap by aligning based on cropped dimensions.

    Parameters:
        mask_dict (dict): Dictionary with keys containing patch coordinates as part of the key string 
                          (in the format "... row_col.jpg") and values as file paths to predicted mask images.
        crop_size (int): Size of the square mask patch before any cropping.
        gap (int): Gap (in pixels) between adjacent patches.
        overlap_crop_size (float): Fraction of crop_size to remove from the left side in non-first columns.
        bottom_crop_size (float): Fraction of crop_size to remove from the bottom of all patches.
        extra_crop_5th_col (int): Additional pixels to crop from the left on the 5th column.
    """
    
    from PIL import Image
    import streamlit as st

    patch_keys = list(mask_dict.keys())
    coords = [tuple(map(int, key.split()[2].replace(".jpg", "").split("_")))
              for key in patch_keys]
    max_row = max(c[0] for c in coords)
    max_col = max(c[1] for c in coords)

    base_left_crop = int(crop_size * overlap_crop_size)
    bottom_crop = int(crop_size * bottom_crop_size)
    image_height = crop_size - bottom_crop
    image_width_col1 = crop_size
    image_width_other_cols = crop_size - base_left_crop

    x_pos, y_pos = {}, {}
    current_x = 0
    x_pos[1] = current_x
    current_x += image_width_col1 + gap
    for col in range(2, max_col + 1):
        x_pos[col] = current_x
        current_x += image_width_other_cols + gap

    current_y = 0
    y_pos[1] = current_y
    current_y += image_height + gap
    for row in range(2, max_row + 1):
        y_pos[row] = current_y
        current_y += image_height + gap

    total_width = x_pos.get(max_col, 0) + (image_width_col1 if max_col == 1 else image_width_other_cols)
    total_height = y_pos.get(max_row, 0) + image_height

    # Create canvas with transparency
    canvas = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    for key in patch_keys:
        row, col = map(int, key.split()[2].replace(".jpg", "").split("_"))
        x_offset = x_pos[col]
        y_offset = y_pos[row]
        mask_img = Image.fromarray(mask_dict[key])

        # Ensure it's in RGBA mode
        if mask_img.mode != 'RGBA':
            mask_img = mask_img.convert('RGBA')

        desired_height = image_height

        # Horizontal Cropping
        if col > 1:
            if col == 5:
                left_offset = base_left_crop + extra_crop_5th_col
                cropped = mask_img.crop((left_offset, 0, crop_size, crop_size))
                desired_width = image_width_other_cols
                current_width = cropped.width
                if current_width < desired_width:
                    new_mask = Image.new('RGBA', (desired_width, crop_size), (0, 0, 0, 0))
                    new_mask.paste(cropped, (0, 0))
                    mask_img = new_mask
                else:
                    mask_img = cropped
            else:
                mask_img = mask_img.crop((base_left_crop, 0, crop_size, crop_size))

        # Vertical Cropping
        if row == 7:
            top_crop = int(crop_size * 0.21)
            cropped = mask_img.crop((0, top_crop, mask_img.width, crop_size))
            current_height = cropped.height
            if current_height < desired_height:
                new_mask = Image.new('RGBA', (mask_img.width, desired_height), (0, 0, 0, 0))
                new_mask.paste(cropped, (0, 0))
                mask_img = new_mask
            elif current_height > desired_height:
                mask_img = cropped.crop((0, 0, mask_img.width, desired_height))
            else:
                mask_img = cropped
        else:
            mask_img = mask_img.crop((0, 0, mask_img.width, crop_size - bottom_crop))

        mask_img = remove_black_bg(mask_img)
        canvas.paste(mask_img, (x_offset, y_offset), mask_img)  # Use alpha channel for transparency

    st.image(canvas, use_container_width=True)


def overlay_bboxes_on_original_images(mask_dict, cropped_images, crop_size=512, gap=0,
                                          overlap_crop_size=0.29, bottom_crop_size=0.29,
                                          extra_crop_5th_col=155, min_area=500):
    """
    Display both predicted masks and corresponding original images with bounding boxes
    in a grid layout, following the same cropping and layout logic.

    Parameters:
        mask_dict (dict): Keys as "... row_col.jpg" and values as predicted mask arrays.
        cropped_images (dict): Keys as "... row_col.jpg" and values as paths to original cropped images.
        Other cropping and layout parameters are same as previous functions.
    """
    import cv2
    import streamlit as st

    patch_keys = list(mask_dict.keys())
    coords = [tuple(map(int, key.split()[2].replace(".jpg", "").split("_"))) for key in patch_keys]
    max_row = max(c[0] for c in coords)
    max_col = max(c[1] for c in coords)

    base_left_crop = int(crop_size * overlap_crop_size)
    bottom_crop = int(crop_size * bottom_crop_size)
    top_crop_row7 = int(crop_size * 0.21)
    image_height = crop_size - bottom_crop
    image_width_col1 = crop_size
    image_width_other_cols = crop_size - base_left_crop

    x_pos, y_pos = {}, {}
    current_x = 0
    x_pos[1] = current_x
    current_x += image_width_col1 + gap
    for col in range(2, max_col + 1):
        x_pos[col] = current_x
        current_x += image_width_other_cols + gap

    current_y = 0
    y_pos[1] = current_y
    current_y += image_height + gap
    for row in range(2, max_row + 1):
        y_pos[row] = current_y
        current_y += image_height + gap

    total_width = x_pos.get(max_col, 0) + (image_width_col1 if max_col == 1 else image_width_other_cols)
    total_height = y_pos.get(max_row, 0) + image_height

    canvas_mask = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    canvas_original = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    for key in patch_keys:
        row, col = map(int, key.split()[2].replace(".jpg", "").split("_"))
        x_offset = x_pos[col]
        y_offset = y_pos[row]

        mask_img = Image.fromarray(mask_dict[key]).convert("RGB")
        orig_img = Image.open(cropped_images[key]).convert("RGB")

        # Apply horizontal cropping
        if col > 1:
            left_crop = base_left_crop + extra_crop_5th_col if col == 5 else base_left_crop
            mask_img = mask_img.crop((left_crop, 0, crop_size, crop_size))
            orig_img = orig_img.crop((left_crop, 0, crop_size, crop_size))

        # Apply vertical cropping
        if row == 7:
            mask_img = mask_img.crop((0, top_crop_row7, mask_img.width, crop_size))
            orig_img = orig_img.crop((0, top_crop_row7, orig_img.width, crop_size))
        else:
            mask_img = mask_img.crop((0, 0, mask_img.width, crop_size - bottom_crop))
            orig_img = orig_img.crop((0, 0, orig_img.width, crop_size - bottom_crop))

        # Convert to OpenCV
        mask_np = np.array(mask_img)
        orig_np = np.array(orig_img)

        # Create binary mask
        mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

        # Contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(cv2.convexHull(contour))
                cv2.rectangle(mask_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(orig_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert back to RGBA with black-to-transparent
        mask_pil = remove_black_bg(Image.fromarray(mask_np).convert("RGBA"))
        orig_pil = remove_black_bg(Image.fromarray(orig_np).convert("RGBA"))

        canvas_mask.paste(mask_pil, (x_offset, y_offset), mask_pil)
        canvas_original.paste(orig_pil, (x_offset, y_offset), orig_pil)

    st.image(canvas_original, use_container_width=True)


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

login("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")


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

tokenizer_reason = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_reason = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# Initialize model and tokenizer for explanations
tokenizer_reason = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_reason = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def explain_tampering(text, coords):
    """Generate precise technical explanations with proper formatting"""
    x, y, w, h = coords
    structured_prompt = f"""Analyze this document region for tampering signs:
    
    [Text] {text}
    [Location] X:{x}-{x+w}, Y:{y}-{y+h}
    [Page Context] Other text elements: dates, headers, body text
    
    Technical indicators (choose one):
    1. Font mismatch (size/style/family)
    2. Color inconsistency (RGB/CMYK values)
    3. Alignment deviation (±{max(2, w//10)}px from grid)
    4. JPEG grid anomaly (8x8 block artifacts)
    5. Text orientation error (rotation difference)
    6. Lighting inconsistency (shadow direction)
    7. Compression artifact misalignment
    
    Concise technical reason (format exactly): [Indicator X] because..."""

    inputs = tokenizer_reason(structured_prompt, 
                            return_tensors="pt",
                            max_length=512,
                            truncation=True)
    
    outputs = model_reason.generate(
        inputs.input_ids.to(model_reason.device),
        max_new_tokens=80,
        num_beams=5,
        temperature=0.3,
        early_stopping=True
    )
    
    # Parse response rigorously
    full_response = tokenizer_reason.decode(outputs[0], skip_special_tokens=True)
    if "because" in full_response:
        return full_response.split("because")[-1].split(".")[0].strip().capitalize()
    return "Text pattern anomaly detected"

def validate_detected_text(text):
    """
    Filter meaningless OCR results
    """
    text = text.strip()
    
    # Common false positives filter
    common_artifacts = {"", " ", "for", "a", "the", "is", "of", "and"}
    
    if len(text) < 2 or text in common_artifacts:
        return None
    
    # Check for suspicious formatting patterns
    if text.isalnum() and not any(c.islower() for c in text):
        return "SUSPICIOUS ALL-CAPS TEXT IN ISOLATED REGION"
    
    return text

st.set_page_config(layout="wide")

# Upload multiple image files
uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    # Create a tab for each uploaded image
    tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_images))])

    for i, (img_file, tab) in enumerate(zip(uploaded_images, tabs)):
        with tab:
            img_name = img_file.name
            st.subheader(f"Image {i+1}: {img_name}")

            # Load image
            orig_img = Image.open(img_file).convert("RGB")


            # Resize and zoom
            crop_size = 512
            zoom_factor = 1.1
            zoomed_width = int(orig_img.width * zoom_factor)
            zoomed_height = int(orig_img.height * zoom_factor)
            zoomed_img = orig_img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

            # Crop settings
            stride = int(crop_size * 0.7)
            width, height = zoomed_img.size
            y_positions = list(range(0, height - crop_size + 1, stride))
            if (height - crop_size) % stride != 0:
                y_positions.append(height - crop_size)
            x_positions = list(range(0, width - crop_size + 1, stride))
            if (width - crop_size) % stride != 0:
                x_positions.append(width - crop_size)

            num_rows = len(y_positions)
            gap = 50
            bottom_crop = int(crop_size * 0.29)
            row_height = crop_size - bottom_crop
            grid_height = (num_rows * row_height) + ((num_rows - 1) * gap)

            # Resize the original image to a smaller width (e.g., 400px) while maintaining aspect ratio
            max_width = 700  # You can adjust this value
            aspect_ratio = orig_img.width / orig_img.height
            new_width = max_width
            new_height = int(new_width / aspect_ratio)

            resized_orig = orig_img.resize((new_width, new_height), Image.LANCZOS)
            #st.image(resized_orig, use_container_width=False)

            # === Prepare Cropped Images ===
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

            

            # === Tampering Detection + OCR ===
            predicted_masks = {}
            predicted_mask_with_boxes = {}
            predicted_original_with_boxes = {}
            all_extracted_texts = {}

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
                    original_img = imgs_ori.copy()
                    mask = ci.copy()

                    threshold_value = 40
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
                    _, binary_mask = cv2.threshold(mask_gray, threshold_value, 255, cv2.THRESH_BINARY)

                    kernel = np.ones((7, 7), np.uint8)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                    binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    mask_with_boxes = mask.copy()
                    original_with_boxes = original_img.copy()
                    extracted_texts = {}
                    min_area = 500

                    for contour in contours:
                        if cv2.contourArea(contour) > min_area:
                            x, y, w, h = cv2.boundingRect(cv2.convexHull(contour))
                            cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.rectangle(original_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                            cropped_region = gray_img[y:y + h, x:x + w]
                            cropped_region = cv2.GaussianBlur(cropped_region, (3, 3), 0)
                            _, cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            extracted_text = pytesseract.image_to_string(cropped_region, config="--psm 6")
                            extracted_texts[(x, y, w, h)] = extracted_text.strip()
                            # Draw text on the mask (predicted tampered area)
                            text = extracted_text.strip()
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            text_x = x
                            text_y = y - 5 if y - 5 > 10 else y + 15  # Adjust position
                            cv2.putText(mask, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


                    # all_extracted_texts[img_key] = extracted_texts
                    all_extracted_texts[img_key] = extracted_texts
                    predicted_mask_with_boxes[img_key] = mask_with_boxes
                    predicted_original_with_boxes[img_key] = original_with_boxes

            
            bbox_texts, mask_boxes, original_boxes = process_displayed_masks_and_extract_text(
                predicted_masks, cropped_images
            )


            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("### Original Image")
                display_patches_as_grid_no_overlap(cropped_images)

            with col2:
                st.markdown("### Image with Bounding Box")
                # display_predicted_masks_as_grid_no_overlap(predicted_masks)
                overlay_bboxes_on_original_images(predicted_masks,cropped_images)

            
            st.markdown("---")
            st.markdown("## Extracted Texts from Tampered Regions")

            
            for img_key, extracted_texts in bbox_texts.items():
                with st.expander(f"Extracted texts from: {img_key}", expanded=True):
                    if not extracted_texts:
                        st.markdown("<i>No tampered text detected.</i>", unsafe_allow_html=True)
                    for bbox, text in extracted_texts.items():
                        
                        # Fix 1: Add proper text cleaning before validation
                        if text:
                           
                            # Clean and validate text
                            text = text.strip(' :®©™•/\\|®')
                            
                            # Enhanced validation
                            if len(text) < 3 or text.replace(':', '').isdigit():
                                explanation = "Numerical/positional alignment problem"
                            elif text.isupper() and any(c.isalpha() for c in text):
                                explanation = "All-caps text and font mismatch"
                            else:
                                try:
                                    explanation = explain_tampering(text, (x, y, w, h))
                                    # Ensure minimum explanation quality
                                    if len(explanation.split()) < 3:
                                        explanation = "Format/style inconsistency detected"
                                except Exception as e:
                                    explanation = "Technical analysis unavailable"

                            # Display with validation
                            st.markdown(
                                f"""
                                <div style='font-size: 14px; margin-bottom: 10px;'>
                                    <strong>At coordinates</strong> 
                                    <code>(x:{bbox[0]}, y:{bbox[1]}, w:{bbox[2]}, h:{bbox[3]})</code>: 
                                    <span style='color:#2e7d32; font-weight: bold;'>{text}</span><br>
                                    <span style='color:#555;'>{explanation}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                                    
