import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from io import BytesIO
import os
import json
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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

def generate_overlapping_crops(image, crop_size=512, zoom_factor=1.1, overlap_ratio=0.7):
    """
    Generate overlapping cropped patches from an image after zooming.
    """
    # Step 1: Resize (zoom in)
    zoomed_width = int(image.width * zoom_factor)
    zoomed_height = int(image.height * zoom_factor)
    zoomed_img = image.resize((zoomed_width, zoomed_height), Image.LANCZOS)

    # Step 2: Define crop positions
    stride = int(crop_size * overlap_ratio)
    width, height = zoomed_img.size

    y_positions = list(range(0, height - crop_size + 1, stride))
    if (height - crop_size) % stride != 0:
        y_positions.append(height - crop_size)

    x_positions = list(range(0, width - crop_size + 1, stride))
    if (width - crop_size) % stride != 0:
        x_positions.append(width - crop_size)

    # Step 3: Crop and collect patches
    cropped_images = {}
    for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
        box = (x, y, x + crop_size, y + crop_size)
        cropped_image = zoomed_img.crop(box)

        img_io = BytesIO()
        cropped_image.save(img_io, format="JPEG")
        img_io.seek(0)

        row = (idx // len(x_positions)) + 1
        col = (idx % len(x_positions)) + 1
        key = f"{row}_{col}.jpg"
        cropped_images[key] = img_io

    return cropped_images

def crop_black_borders(image, threshold=10):
    """
    Crop black borders from an image based on a pixel intensity threshold.
    """
    img_array = np.asarray(image)
    gray = np.mean(img_array, axis=2)

    mask = gray > threshold  

    if not mask.any():
        return image  

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  

    cropped_array = img_array[y0:y1, x0:x1]
    return Image.fromarray(cropped_array)

def display_patches_as_grid_no_overlap(patches_dict, crop_size=512, gap=0, 
                                      overlap_crop_size=0.29, bottom_crop_size=0.29, 
                                      extra_crop_5th_col=155):
    """
    Combine patches into a grid without overlap and return the final image.
    """
    patch_keys = list(patches_dict.keys())
    coords = [tuple(map(int, key.replace(".jpg", "").split("_"))) for key in patch_keys]
    max_row = max(c[0] for c in coords)
    max_col = max(c[1] for c in coords)

    # Crop calculations
    base_left_crop = int(crop_size * overlap_crop_size)
    bottom_crop = int(crop_size * bottom_crop_size)
    image_height = crop_size - bottom_crop
    image_width_col1 = crop_size
    image_width_other_cols = crop_size - base_left_crop

    # Precompute x/y positions
    x_pos, current_x = {1: 0}, image_width_col1 + gap
    for col in range(2, max_col + 1):
        x_pos[col] = current_x
        current_x += image_width_other_cols + gap

    y_pos, current_y = {1: 0}, image_height + gap
    for row in range(2, max_row + 1):
        y_pos[row] = current_y
        current_y += image_height + gap

    total_width = x_pos.get(max_col, 0) + (image_width_col1 if max_col == 1 else image_width_other_cols)
    total_height = y_pos.get(max_row, 0) + image_height
    canvas = Image.new('RGB', (total_width, total_height), (0, 0, 0))

    for key in patch_keys:
        row, col = map(int, key.replace(".jpg", "").split("_"))
        x_offset = x_pos[col]
        y_offset = y_pos[row]

        # Open image from BytesIO
        patch_img = Image.open(patches_dict[key]).convert("RGB")

        # Horizontal cropping
        if col > 1:
            left_crop = base_left_crop
            if col == 5:
                left_crop += extra_crop_5th_col
            patch_img = patch_img.crop((left_crop, 0, crop_size, crop_size))

        # Vertical cropping
        vertical_crop = crop_size - bottom_crop
        if row == 7:
            top_crop = int(crop_size * 0.21)
            patch_img = patch_img.crop((0, top_crop, patch_img.width, patch_img.height))
            vertical_crop = min(vertical_crop, patch_img.height)
        patch_img = patch_img.crop((0, 0, patch_img.width, vertical_crop))

        # Paste to canvas
        canvas.paste(patch_img, (x_offset, y_offset))

    return crop_black_borders(canvas)

def remove_black_bg(image_rgba):
    """
    Replace pure black background with transparent background.
    """
    data = np.array(image_rgba)
    r, g, b, a = data.T
    black_areas = (r == 0) & (g == 0) & (b == 0)
    data[..., :-1][black_areas.T] = (0, 0, 0)
    data[..., -1][black_areas.T] = 0  # Set alpha to 0
    return Image.fromarray(data)


def overlay_bboxes(bbox_output, cropped_images, crop_size=512, gap=0,
                   overlap_crop_size=0.29, bottom_crop_size=0.29,
                   extra_crop_5th_col=155):
    """Overlay bounding boxes on patches and combine into a grid."""
    if not bbox_output or not cropped_images:
        return Image.new('RGB', (100, 100), (255, 255, 255))
        
    patch_keys = list(bbox_output.keys())
    
    # FIX: Correctly parse keys like '1_1.jpg'
    coords = [tuple(map(int, key.replace(".jpg", "").split("_"))) for key in patch_keys]
    
    if not coords:
        return Image.new('RGB', (100, 100), (255, 255, 255))

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

    canvas_original = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    for key in patch_keys:
        # FIX: Correctly parse the key 'row_col.jpg'
        row, col = map(int, key.replace(".jpg", "").split("_"))
        x_offset = x_pos[col]
        y_offset = y_pos[row]

        # Get original patch image. The key format now matches.
        orig_img = Image.open(cropped_images[key]).convert("RGB")
        orig_width, orig_height = orig_img.size

        # Cropping based on row/column position
        if col > 1:
            left_crop = base_left_crop + (extra_crop_5th_col if col == 5 else 0)
            orig_img = orig_img.crop((left_crop, 0, orig_width, orig_height))
        if row == 7:
            orig_img = orig_img.crop((0, top_crop_row7, orig_img.width, orig_height))
        else:
            orig_img = orig_img.crop((0, 0, orig_img.width, orig_height - bottom_crop))

        orig_np = np.array(orig_img)

        # Draw bounding boxes
        bboxes = bbox_output.get(key, [])
        for box in bboxes:
            if box:
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                cv2.rectangle(orig_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Process and paste to canvas
        orig_pil = Image.fromarray(orig_np).convert("RGBA")
        orig_pil = remove_black_bg(orig_pil)
        canvas_original.paste(orig_pil, (x_offset, y_offset), orig_pil)

    cropped_canvas = crop_black_borders(canvas_original)
    
    if cropped_canvas.mode == 'RGBA':
        cropped_canvas = cropped_canvas.convert('RGB')
        
    return cropped_canvas

def load_bbox_data(json_path):
    """
    Load bounding box data from a JSON file.

    Parameters:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Dictionary of page-wise bounding boxes.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    bbox_output = {
        page: [entry["bbox"] for entry in entries]
        for page, entries in data["result"].items()
    }
    
    return bbox_output
    
@app.route('/', methods=['GET', 'POST'])
def index():
    stitched_image_path = None
    bbox_image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            orig_img = Image.open(file).convert("RGB")
            
            cropped_images = generate_overlapping_crops(orig_img)
            stitched_image = display_patches_as_grid_no_overlap(cropped_images)
            stitched_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
            stitched_image.save(stitched_path)
            stitched_image_path = url_for('static', filename='original.jpg')

            # Load bounding boxes from JSON
            json_path = "/home/xelpmoc/Documents/DocTamperAPI_Draft/flask_app/detect_tampering.json"
            bbox_data_from_json = load_bbox_data(json_path)

            # FIX: Convert keys from 'row.col' to 'row_col.jpg' to match cropped_images
            if bbox_data_from_json:
                bbox_output_corrected = {
                    key.replace('.', '_') + '.jpg': value
                    for key, value in bbox_data_from_json.items()
                }

                bbox_image = overlay_bboxes(bbox_output_corrected, cropped_images)
                bbox_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bboxes.jpg')
                bbox_image.save(bbox_path)
                bbox_image_path = url_for('static', filename='bboxes.jpg')

    return render_template(
        'index.html',
        stitched_image_path=stitched_image_path,
        bbox_image_path=bbox_image_path
    )

if __name__ == '__main__':
    app.run(debug=True)