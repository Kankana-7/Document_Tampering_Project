from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import numpy as np
import time
import torch
import src.model_load  
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import jpegio
import pytesseract
import cv2
import os

app = FastAPI()

@app.on_event("startup")
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, new_qtb, totsr, toctsr = src.model_load.load_segmentation_model(device=device)
        app.state.model = model
        app.state.device = device
        app.state.toctsr = toctsr
        app.state.totsr = totsr
        app.state.new_qtb = new_qtb
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

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

def process_displayed_masks_and_extract_text(predicted_masks, cropped_images,
                                             crop_size=512, overlap_crop_size=0.29,
                                             bottom_crop_size=0.29, extra_crop_5th_col=155):
    """
    Process the cropped & displayed predicted masks, compute bounding boxes,
    and extract text from corresponding regions in original cropped images.

    Returns:
        - bbox_extracted_texts: dict with {key: {bbox_str: text}}
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

                # Use string key for bbox instead of tuple
                bbox_key = f"{x},{y},{w},{h}"
                extracted_texts[bbox_key] = text

                # Annotate text
                cv2.putText(mask_with_boxes, text, (x, y - 5 if y > 10 else y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Store outputs
        bbox_extracted_texts[key] = extracted_texts
        bbox_mask_with_text[key] = mask_with_boxes
        bbox_original_with_text[key] = original_with_boxes

    return bbox_extracted_texts, bbox_mask_with_text, bbox_original_with_text

def display_extracted_texts_with_explanations(bbox_texts, explain_func=None):
    result = {}

    for img_key, extracted_texts in bbox_texts.items():
        img_key = img_key.replace("Cropped Image ", "").replace(".jpg", "").replace("_", ".")
        result[img_key] = []

        if not extracted_texts:
            result[img_key].append({
                "text": "No tampered text detected.",
                "bbox": None,
                "reason": None
            })
            continue

        for bbox_str, text in extracted_texts.items():
            if text:
                text = text.strip(' :®©™•/\\|®')

                try:
                    x, y, w, h = map(int, bbox_str.split(","))
                except Exception:
                    continue

                if len(text) < 3 or text.replace(':', '').isdigit():
                    explanation = "Numerical/positional alignment problem"
                elif text.isupper() and any(c.isalpha() for c in text):
                    explanation = "All-caps text and font mismatch"
                else:
                    try:
                        if explain_func:
                            explanation = explain_func(text, (x, y, w, h))
                            if len(explanation.split()) < 3:
                                explanation = "Format/style inconsistency"
                        else:
                            explanation = "Format/style inconsistency"
                    except Exception:
                        explanation = "Could not analyze text formatting"

                result[img_key].append({
                    "text": text,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "reason": explanation
                })

    return result

def inference_on_image(orig_img: Image.Image, model, device):

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

        # data_tensor = toctsr(input_img).unsqueeze(0).to(device)
        data_tensor = app.state.toctsr(input_img).unsqueeze(0).to(device)
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

@app.post("/detect_tampering")
async def predict_tamper_region(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        # result = inference_on_image(img, app.state.model, app.state.device)
        all_texts = inference_on_image(img, app.state.model, app.state.device)
        result = display_extracted_texts_with_explanations(all_texts)

        processing_time = time.time() - start_time
        return JSONResponse(content={
            "status": "success",
            "result": result,
            "processing_time": f"{processing_time:.2f} seconds"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>Document Tampering Detection System</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #0d0d0d;
                    margin: 0;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .container {
                    background-color: #1e1e1e;
                    border: 2px solid #00ff99;
                    border-radius: 20px;
                    width: 650px;
                    height: 500px;
                    box-shadow: 0 0 20px #00ff99;

                    /* Flexbox to center content vertically and horizontally */
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    padding: 30px 20px;
                }
                h1 {
                    color: #00ff99;
                    font-size: 26px;
                    margin-bottom: 15px;
                    white-space: nowrap;
                }
                p {
                    color: #cccccc;
                    font-size: 18px;
                    margin-bottom: 25px;
                    text-align: center;
                }
                input[type="file"] {
                    display: none;
                }
                label {
                    background-color: #0099ff;
                    color: white;
                    padding: 12px 25px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-bottom: 10px;
                    display: inline-block;
                    transition: background-color 0.3s;
                }
                label:hover {
                    background-color: #007acc;
                }
                #file-name {
                    font-style: italic;
                    font-size: 14px;
                    color: #aaa;
                    margin-bottom: 20px;
                    text-align: center;
                }
                input[type="submit"] {
                    background-color: #ff4081;
                    border: none;
                    color: white;
                    padding: 14px 30px;
                    font-size: 18px;
                    border-radius: 5px;
                    cursor: pointer;
                    box-shadow: 0 0 12px #ff4081;
                    transition: background-color 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #e91e63;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📄 Document Tampering Detection System</h1>
                <p>Please upload an image file to detect tampering...</p>
                <form action="/detect_tampering" enctype="multipart/form-data" method="post" style="display:flex; flex-direction:column; align-items:center;">
                    <label for="file-upload">Choose Image</label>
                    <input id="file-upload" name="file" type="file" accept="image/*" onchange="showFileName()">
                    <div id="file-name">No file chosen</div>
                    <input type="submit" value="Detect Tampering">
                </form>
            </div>
            <script>
                function showFileName() {
                    const input = document.getElementById('file-upload');
                    const fileNameDisplay = document.getElementById('file-name');
                    if (input.files.length > 0) {
                        fileNameDisplay.textContent = input.files[0].name;
                    } else {
                        fileNameDisplay.textContent = 'No file chosen';
                    }
                }
            </script>
        </body>
    </html>
    """

import uvicorn
from pyngrok import ngrok

# # Set your ngrok auth token (replace with your real token)
# ngrok.set_auth_token("2xzCCP6VkidD5aUYCnP2umNbOsw_5Smd8Er1QeQNopLhjnqrL")

# # Open a tunnel on port 8000
# public_url = ngrok.connect(8000)
# print(f"🌐 Public URL: {public_url}")

# # Run the FastAPI app
# uvicorn.run("app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Set ngrok auth token
    ngrok.set_auth_token("2xzCCP6VkidD5aUYCnP2umNbOsw_5Smd8Er1QeQNopLhjnqrL")
    
    # Open ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"🌐 Public URL: {public_url}")
    
    # Run FastAPI app
    uvicorn.run("main_final:app", host="0.0.0.0", port=8000)