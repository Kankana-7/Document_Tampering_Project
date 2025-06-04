from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from huggingface_hub import login

login("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")


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

def display_extracted_texts_with_explanations(bbox_texts, explain_func=None):
    """
    Processes extracted text from bounding boxes and attaches simple explanations.

    Args:
        bbox_texts (dict): Dictionary where keys are image identifiers and values are dicts of {bbox: text}.
        explain_func (callable): Optional custom function for detailed explanation. Signature: explain_func(text, bbox).

    Returns:
        dict: A structured dictionary with image keys and a list of entries containing text, bbox, and explanation.
    """
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

        for bbox, text in extracted_texts.items():
            if text:
                text = text.strip(' :®©™•/\\|®')

                if len(text) < 3 or text.replace(':', '').isdigit():
                    explanation = "Numerical/positional alignment problem"
                elif text.isupper() and any(c.isalpha() for c in text):
                    explanation = "All-caps text and font mismatch"
                else:
                    try:
                        if explain_func:
                            explanation = explain_func(text, bbox)
                            if len(explanation.split()) < 3:
                                explanation = "Format/style inconsistency"
                        else:
                            explanation = "Format/style inconsistency"
                    except Exception:
                        explanation = "Could not analyze text formatting"

                result[img_key].append({
                    "text": text,
                    # "bbox": bbox,
                    "bbox": [{"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}],
                    "reason": explanation
                })

    return result
