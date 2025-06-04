import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from albumentations import ToTensorV2
import torchvision
import cv2
import jpegio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import io
from io import BytesIO
from tempfile import NamedTemporaryFile
from PIL import ImageOps
import asyncio
from PIL import Image
from io import BytesIO
from src.dtd import seg_dtd
from src.fph import FPH
from src.model_load import *
from src.image_processing import *
from src.predict import *
from src.reasons import *
# from src.inference_pipeline import *
from src.inf import *
from src.swins import *
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import threading
import json
import tempfile

app = FastAPI()

# Load the model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, new_qtb, totsr, toctsr = load_segmentation_model(device=device)

@app.post("/detect_tampering")    
async def predict_tamper_region(file: UploadFile = File(...)):
    try:
        # Read image bytes and convert to PIL Image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Inference
        all_texts = inference_on_image(img, model, device)
        result = display_extracted_texts_with_explanations(all_texts)

        return JSONResponse(content={"status": "success", "result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
