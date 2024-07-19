from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
import spacy
from paddleocr import PaddleOCR
from langdetect import detect
import json
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREPROCESSED_FOLDER'] = 'static/preprocessed'

# Load models
nlp_en = spacy.load("en_core_web_sm")

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREPROCESSED_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    preprocessed_path = image_path.replace('uploads', 'preprocessed')
    cv2.imwrite(preprocessed_path, denoised)
    return preprocessed_path

def advanced_ocr(image_path):
    result = ocr.ocr(image_path, cls=True)
    text = ""
    bboxes = []
    for line in result:
        for word in line:
            text += word[1][0] + " "
            bboxes.append(word[0])
        text += "\n"
    return text, bboxes

def extract_entities_from_text(text):
    data = {
        "name": "",
        "company": "",
        "job_title": "",
        "phone": "",
        "email": "",
        "address": ""
    }

    # Define regex patterns for different fields
    patterns = {
        "name": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b",
        "company": r"\b(?:Inc|Ltd|Corporation|Corp|LLC)\b",
        "job_title": r"\b(Manager|Director|Engineer|Developer|Specialist|General Manager)\b",
        "phone": r"\+?\d[\d -]{7,12}\d",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b",
        "address": r"\d+\s+\w+\s+\w+"
    }

    # Match patterns and fill data
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            if key in ["phone", "email"]:
                data[key] = matches[0]
            else:
                data[key] = matches[0]

    # Use NLP for more robust extraction
    doc = nlp_en(text)
    for entity in doc.ents:
        if entity.label_ == "PERSON" and not data['name']:
            data['name'] = entity.text
        elif entity.label_ == "ORG" and not data['company']:
            data['company'] = entity.text
        elif entity.label_ in ("GPE", "LOC") and not data['address']:
            data['address'] = entity.text
        elif entity.label_ == "EMAIL" and not data['email']:
            data['email'] = entity.text
        elif entity.label_ == "PHONE" and not data['phone']:
            data['phone'] = entity.text
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        preprocessed_path = preprocess_image(filepath)
        extracted_text, bboxes = advanced_ocr(preprocessed_path)
        entities = extract_entities_from_text(extracted_text)

        return render_template('edit.html', entities=entities)

@app.route('/download', methods=['POST'])
def download_json():
    data = request.form.to_dict(flat=False)
    json_data = json.dumps(data, indent=4)
    path = 'static/output.json'
    with open(path, 'w') as f:
        f.write(json_data)
    return send_file(path, as_attachment=True, attachment_filename='output.json')

@app.context_processor
def utility_processor():
    def is_list(value):
        return isinstance(value, list)
    return dict(is_list=is_list)

if __name__ == '__main__':
    app.run(debug=True)