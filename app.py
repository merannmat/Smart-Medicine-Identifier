from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
import os

from utils.ocr import (
    extract_text_from_image,
    extract_medicine_name
)

from utils.matcher import find_medicine
from utils.gemini_search import search_medicine_online
from utils.update_dataset import save_new_medicine
from utils.rebuild_index import rebuild_index



app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image uploaded'
        })

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No file selected'
        })

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ---------------- OCR ----------------

    medicine_name = extract_text_from_image(filepath)

    if not medicine_name:
        return jsonify({
            'success': False,
            'message': 'Could not extract text from image'
        })

    print()
    print("===== OCR TEXT =====")
    print(medicine_name)
    print()

    # ------------ Local Search ------------

    result = find_medicine(medicine_name)

    if result:

        print("Medicine found locally.")
        print()

        return jsonify({
            'success': True,
            'source': 'local',

            'medicine_name': result['medicine_name'],
            'composition': result['composition'],
            'uses': result['uses'],
            'side_effects': result['side_effects'],
            'manufacturer': result['manufacturer'],
            'confidence': result['confidence']
        })

    # ------------ Gemini Fallback ------------

    print("Medicine not found locally.")
    print("Searching online with Gemini...")
    print()

    online_result = search_medicine_online(medicine_name)

    if online_result:

        print("Medicine found using Gemini.")
        
        #only save if gemini is reasonably sure 
        if online_result['medicine_name'] != "Unknown":
            added = save_new_medicine(online_result)

            if added:
                rebuild_index()

        return jsonify({
            'success': True,
            'source': 'gemini',

            'medicine_name': online_result.get(
                'medicine_name',
                 medicine_name
            ),

            'composition': online_result.get(
                'composition',
                'Unknown'
            ),

            'uses': online_result.get(
                'uses',
                'Unknown'
            ),

            'side_effects': online_result.get(
                'side_effects',
                'Unknown'
            ),

            'manufacturer': online_result.get(
                'manufacturer',
                'Unknown'
            ),

            'confidence': 75
        })

    # ------------ Complete Failure ------------

    return jsonify({
        'success': False,
        'medicine_name': medicine_name,
        'message': 'Medicine not found locally or online'
    })


if __name__ == '__main__':
    app.run(debug=True)