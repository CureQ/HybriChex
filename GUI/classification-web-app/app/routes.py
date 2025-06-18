from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
import os
from app.model.model import model  # Import the ParallelCheXNetSwin model
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
import uuid

# Define the blueprint
main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'


ALLOWED_EXTENSIONS = {'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define routes within the blueprint
@main.route('/')
def index():
    return render_template('dashboard.html')

@main.route('/predict', methods=['POST'])
def predict():
    return jsonify({'error': 'Predict functionality is not implemented'}), 501

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.png'):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return f"File {file.filename} uploaded successfully", 200
    return "Invalid file type", 400

@main.route('/documentation')
def documentation():
    return render_template('documentation.html')

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/model-classifier', methods=['GET', 'POST'])
def model_classifier():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            session['uploaded_image'] = file_path  # Store the uploaded image path in the session
            return redirect(url_for('main.radiologist_classifier'))
    uploaded_image = session.get('uploaded_image', None)
    return render_template('model_classifier.html', uploaded_image=uploaded_image)


@main.route('/radiologist-classifier')
def radiologist_classifier():
    uploaded_image = session.get('uploaded_image', None)
    model_classifications = session.get('model_classifications', [])
    if uploaded_image:
        uploaded_image = f"/static/uploads/{os.path.basename(uploaded_image)}"  # Convert to static path
    return render_template('radiologist_classifier.html', uploaded_image=uploaded_image, model_classifications=model_classifications)

@main.route('/report')
def report():
    # Sample data to pass to the report template
    report_data = {
        "patient_name": "L. Versteeg",
        "dob": "12-05-1978",
        "exam_type": "Chest X-ray (PA & Lateral)",
        "exam_date": "24-03-2025",
        "clinical_info": "Patient presents with persistent cough, mild fever, and shortness of breath for the past 10 days. Clinical suspicion of lower respiratory infection.",
        "findings": [
            "Increased opacity is noted in the right lower lung field, consistent with an infiltrative process.",
            "No significant pleural effusion or pneumothorax is detected.",
            "Cardiac silhouette and mediastinum appear normal.",
            "No signs of acute bony abnormalities."
        ],
        "model_classification": "Pulmonary Infiltration Detected",
        "radiologist_conclusion": "Findings are consistent with a localized pulmonary infection or early-stage pneumonia. Clinical correlation is advised. Recommend follow-up imaging if symptoms persist or worsen.",
        "radiologist_name": "Dr. Jim Brown, MD",
        "report_date": "24-03-2025"
    }
    return render_template('report.html', report_data=report_data)

from app.model.classifier import Classifier

classifier = Classifier()

@main.route('/classify-image', methods=['POST'])
def classify_image():
    print("🔵 Route '/classify-image' aangeroepen")  # Debug

    if 'file' not in request.files:
        print("❌ Geen bestand gevonden in het formulier")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("❌ Bestand is geselecteerd, maar heeft geen naam")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        print(f"\U0001F4F8 Bestand ontvangen: {file.filename}")
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        # Store uploaded image path in session for later use
        session['uploaded_image'] = file_path
        print(f"\U0001F4BE Bestand opgeslagen op: {file_path}")

        try:
            image = Image.open(file_path).convert('RGB')
            print("\U0001F5BC️ Afbeelding succesvol geopend")

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
            print("\U0001F501 Transformatie toegepast op afbeelding")

            prediction = classifier.classify(image_tensor)
            print(f"\u2705 Voorspelling gegenereerd: {prediction}")

            # Store all positive model classifications in session (as a list)
            if prediction['classifications']:
                session['model_classifications'] = [name for name, prob in prediction['classifications']]
            else:
                session['model_classifications'] = []

            prediction_list = [
                {"class_name": name, "probability": float(prob)}
                for name, prob in prediction['classifications']
            ]
            top5_list = [
                {"class_name": name, "probability": float(prob)}
                for name, prob in prediction['top5']
            ]

            gradcam_url = None
            try:
                # Gebruik alleen de hoogste voorspelde klasse voor GradCAM
                if prediction['top5']:
                    # Pak de index van de hoogste voorspelde klasse
                    probs = [prob for _, prob in prediction['top5']]
                    class_names = [name for name, _ in prediction['top5']]
                    # Zoek de index van de hoogste kans in de originele volgorde
                    top_class_name = class_names[0]
                    top_class_idx = classifier.class_names.index(top_class_name)
                    grayscale_cam = classifier.generate_gradcam(image_tensor, top_class_idx)
                    
                    # Gebruik de originele afbeelding als numpy array (0-255, RGB)
                    input_image = np.array(image).astype(np.float32)
                    if input_image.max() > 1.0:
                        input_image = input_image / 255.0
                    # --- Fix: resize gradcam naar originele afbeelding ---
                    if grayscale_cam.shape != input_image.shape[:2]:
                        import cv2
                        grayscale_cam = cv2.resize(grayscale_cam, (input_image.shape[1], input_image.shape[0]))

                    cam_image = classifier.gradcam_overlay(input_image, grayscale_cam, use_rgb=True)

                    gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
                    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
                    cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                    gradcam_url = f"/static/uploads/{gradcam_filename}"
                    print(f"GradCAM saved at: {gradcam_path}")
                    print(f"GradCAM URL: {gradcam_url}")
                else:
                    gradcam_url = None
            except Exception as e:
                print(f"\u2757Fout bij Grad-CAM generatie: {e}")
                gradcam_url = None
            # --- Einde Grad-CAM generatie ---

            return jsonify({'prediction': prediction_list, 'top5': top5_list, 'gradcam_url': gradcam_url}), 200

        except Exception as e:
            import traceback
            print(f"\u2757Fout tijdens verwerking afbeelding: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Internal server error: {e}'}), 500

    print("❌ Ongeldig bestandstype")
    return jsonify({'error': 'Invalid file type'}), 400

@main.route('/settings')
def settings():
    return render_template('settings.html')