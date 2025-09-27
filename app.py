from flask import Flask, request, jsonify, render_template, url_for
import requests
from PIL import Image
from io import BytesIO
import os
import supervision as sv
import numpy as np
import base64
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename
import traceback
from utils.model_utils import validate_model, process_image
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = "model/train5best.pt"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global model variable
model = None
model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load YOLO model with comprehensive error handling"""
    global model, model_loaded
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False, f"Model file not found at {MODEL_PATH}"
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        
        # Validate model
        is_valid, validation_msg = validate_model(model)
        if not is_valid:
            return False, validation_msg
            
        model_loaded = True
        logger.info(f"Model loaded successfully! Classes: {list(model.names.values())}")
        return True, "Model loaded successfully"
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def run_inference(image_input, is_url=False):
    """
    Run YOLO inference on image
    Args:
        image_input: Either URL string or PIL Image
        is_url: Boolean indicating if input is URL
    Returns:
        tuple: (success, data_or_error_msg)
    """
    try:
        # Load image
        if is_url:
            try:
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except requests.exceptions.Timeout:
                return False, "Request timeout - URL took too long to respond"
            except requests.exceptions.RequestException as e:
                return False, f"Failed to download image: {str(e)}"
        else:
            image = image_input
            
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Process image
        result = process_image(model, image)
        if not result['success']:
            return False, result['error']
            
        return True, result
        
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

# Initialize model on startup
logger.info("Initializing application...")
success, msg = load_model()
if not success:
    logger.error(f"Failed to load model: {msg}")
else:
    logger.info(msg)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         model_classes=list(model.names.values()) if model_loaded and hasattr(model, 'names') else [])

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server logs.',
                'timestamp': datetime.now().isoformat()
            }), 500
            
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
            
        image_url = data.get('image_url')
        confidence_threshold = float(data.get('confidence_threshold', 0.5))
        
        if not image_url:
            return jsonify({
                'success': False,
                'error': 'No image_url provided',
                'timestamp': datetime.now().isoformat()
            }), 400
            
        # Run inference
        success, result = run_inference(image_url, is_url=True)
        
        if not success:
            return jsonify({
                'success': False,
                'error': result,
                'timestamp': datetime.now().isoformat()
            }), 400
            
        # Filter by confidence
        filtered_predictions = [p for p in result['predictions'] 
                              if p['confidence'] >= confidence_threshold]
        
        return jsonify({
            'success': True,
            'detection_count': len(filtered_predictions),
            'total_detections': len(result['predictions']),
            'annotated_image': result['annotated_image'],
            'predictions': filtered_predictions,
            'confidence_threshold': confidence_threshold,
            'model_info': {
                'classes': list(model.names.values()) if hasattr(model, 'names') else [],
                'model_path': MODEL_PATH
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Prediction API error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please refresh the page and try again.'
            }), 500
            
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
            
        file = request.files['file']
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
        # Open and validate image
        try:
            image = Image.open(file.stream)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400
            
        # Run inference
        success, result = run_inference(image, is_url=False)
        
        if not success:
            return jsonify({
                'success': False,
                'error': result
            }), 400
            
        # Filter by confidence
        filtered_predictions = [p for p in result['predictions'] 
                              if p['confidence'] >= confidence_threshold]
        
        return jsonify({
            'success': True,
            'detection_count': len(filtered_predictions),
            'total_detections': len(result['predictions']),
            'annotated_image': result['annotated_image'],
            'predictions': filtered_predictions,
            'confidence_threshold': confidence_threshold,
            'filename': secure_filename(file.filename),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat(),
        'model_classes': list(model.names.values()) if model_loaded and hasattr(model, 'names') else []
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
