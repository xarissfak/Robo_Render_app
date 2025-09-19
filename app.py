from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
from io import BytesIO
import os
import supervision as sv
import numpy as np
import base64
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

# Get API key from environment variable for Render deployment
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    print("ROBOFLOW_API_KEY not found in environment variables.")
    client = None
else:
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    print("Roboflow client initialized successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    if not client:
        return jsonify({"error": "Roboflow API key not configured."}), 500

    data = request.json
    if not data or 'image_url' not in data:
        return jsonify({"error": "Invalid request. Please provide 'image_url'."}), 400

    image_url = data['image_url']

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Image retrieval failed: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

    # Replace with your workflow details
    try:
        result = client.run_workflow(
            workspace_name="cfu-counter",
            workflow_id="custom-workflow-4",
            images={
                "image": image_url
            },
            use_cache=True
        )
    except Exception as e:
        return jsonify({"error": f"Roboflow workflow failed: {e}"}), 500


    processed_predictions = []
    if result and result[0] and 'predictions' in result[0]:
        predictions = result[0]['predictions']['predictions']
        for prediction in predictions:
            processed_predictions.append({
                'x': prediction.get('x'),
                'y': prediction.get('y'),
                'width': prediction.get('width'),
                'height': prediction.get('height'),
                'class': prediction.get('class'),
                'confidence': prediction.get('confidence')
            })

    # Convert processed_predictions to sv.Detections (xyxy)
    detections_list = []
    for p in processed_predictions:
        x_center, y_center, width, height = p['x'], p['y'], p['width'], p['height']
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        detections_list.append([float(x_min), float(y_min), float(x_max), float(y_max), p['confidence'], p['class']])

    if detections_list:
        xyxy = np.array(detections_list)[:, :4].astype(float)
        confidence_values = np.array(detections_list)[:, 4].astype(float)
        class_names = [str(p['class']) for p in processed_predictions]
        class_ids = np.array([0] * len(class_names))

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence_values,
            class_id=class_ids,
            data={'class_name': np.array(class_names)}
        )

        # Create BoxAnnotator
        box_annotator = sv.BoxAnnotator(color=sv.Color.BLACK, thickness=2)

        # Annotate the image
        annotated_image_np = box_annotator.annotate(
            scene=np.array(image),
            detections=detections,
        )

        # Convert annotated image to base64 string
        buffered = BytesIO()
        plt.imshow(annotated_image_np)
        plt.axis('off')
        plt.savefig(buffered, format="PNG")
        plt.close() # Close the plot to prevent displaying it
        img_str = base64.b64encode(buffered.getvalue()).decode()

    else:
        # If no detections, return the original image as base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()


    return jsonify({
        "detection_count": len(processed_predictions),
        "annotated_image": img_str
    })

if __name__ == '__main__':
    # Use Gunicorn for production deployment
    # app.run(debug=True) # For local testing
    pass # Gunicorn will run the app
