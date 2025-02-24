from flask import Flask, request, jsonify
import os
import cv2
import re
import easyocr
import base64
import requests
import multiprocessing
import time
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from io import BytesIO
import uuid
from flask import url_for

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en', 'hi'], gpu=True)

def enhance_image(image):
    """Optimized image preprocessing pipeline"""
    # Resize to improve processing speed
    h, w = image.shape[:2]
    new_width = 800
    new_height = int((new_width / w) * h)
    resized = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # Simple thresholding for faster processing
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

def ocr_processing(image):
    """Process image with optimized OCR pipeline"""
    try:
        enhanced = enhance_image(image)
        results = reader.readtext(enhanced, paragraph=False)
        
        # Early termination checks
        aadhaar_number = False
        aadhaar_text = False
        
        for _, text, _ in results:
            # Check for Aadhaar number pattern
            if re.search(r'\d{4}\s?\d{4}\s?\d{4}', text):
                aadhaar_number = True
                if aadhaar_text:  # Early exit if both found
                    return True
            
            # Check for Aadhaar text markers
            if 'aadhar' in text.lower() or 'आधार' in text.lower():
                aadhaar_text = True
                if aadhaar_number:  # Early exit if both found
                    return True
        
        return aadhaar_number and aadhaar_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return False

def genai_verification(image):
    """Optimized GenAI verification with image resizing"""
    try:
        # Resize image for faster processing
        h, w = image.shape[:2]
        new_width = 800
        new_height = int((new_width / w) * h)
        resized = cv2.resize(image, (new_width, new_height))
        
        # Encode directly to base64
        _, buffer = cv2.imencode('.jpg', resized)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # API call parameters
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this an Aadhaar card? Respond with {'isAadhaar': true/false}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }]}
        

        # Fast timeout with retries
        for _ in range(2):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()['choices'][0]['message']['content']
                    return 'true' in result.lower()
                break
            except requests.Timeout:
                continue
                
        return False
    except Exception as e:
        print(f"GenAI Error: {e}")
        return False

def process_pdf(pdf_bytes):
    """Process PDF with parallel pipelines"""
    try:
        # Convert PDF to images with lower DPI
        images = convert_from_bytes(pdf_bytes, dpi=150, fmt='jpeg')
        image_arrays = [np.array(img) for img in images]

        # Phase 1: Parallel OCR processing
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            ocr_results = pool.map(ocr_processing, image_arrays)

        # Identify images needing GenAI verification
        needs_genai = [
            (idx, img_arr) 
            for idx, (result, img_arr) in enumerate(zip(ocr_results, image_arrays)) 
            if not result
        ]

        # Phase 2: Parallel GenAI processing
        genai_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(genai_verification, img_arr): idx 
                for idx, img_arr in needs_genai
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                genai_results[idx] = future.result()

        # Combine results
        final_results = []
        for idx in range(len(image_arrays)):
            if ocr_results[idx]:
                final_results.append({"page": idx+1, "is_aadhaar": True})
            else:
                final_results.append({
                    "page": idx+1, 
                    "is_aadhaar": genai_results.get(idx, False)
                })
                
        return final_results
    except Exception as e:
        print(f"PDF Processing Error: {e}")
        raise

@app.route('/detect_aadhaar', methods=['POST'])
def detect_aadhaar_endpoint():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    start_time = time.time()
    results = []

    try:
        for file in files:
            filename = file.filename
            unique_filename = f"{uuid.uuid4().hex}_{filename}"  # Unique file name

            if filename.lower().endswith('.pdf'):
                # Process PDF
                pdf_bytes = file.read()
                pdf_results = process_pdf(pdf_bytes)

                # Save PDF pages as images
                images = convert_from_bytes(pdf_bytes, dpi=150, fmt='jpeg')
                for i, img in enumerate(images):
                    image_filename = f"{uuid.uuid4().hex}_page_{i+1}.jpg"
                    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
                    img.save(image_path)  # Save each PDF page as an image

                    # Generate URL
                    image_url = url_for('static', filename=f"uploads/{image_filename}", _external=True)
                    pdf_results[i]["image_url"] = image_url  # Add image URL

                results.extend(pdf_results)

            else:
                # Process individual image
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save uploaded image
                image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                cv2.imwrite(image_path, img)

                # Generate Image URL
                image_url = url_for('static', filename=f"uploads/{unique_filename}", _external=True)

                # Two-stage verification
                ocr_result = ocr_processing(img)
                is_aadhaar = ocr_result or genai_verification(img)
                
                results.append({
                    "filename": filename,
                    "is_aadhaar": is_aadhaar,
                    "image_url": image_url  # Include Image URL
                })

        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f}s")
        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Error processing documents"
        }), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)