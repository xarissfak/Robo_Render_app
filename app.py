from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Roboflow client (���� �� ���� ��� API Key ���)
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="qDv1m9dDwQQrex013Rce"
)

@app.route("/", methods=["GET"])
def home():
    return "?? Roboflow detection API is running!"

@app.route("/detect", methods=["POST"])
def detect():
    # ������� ���� url ���� ������ �������
    if request.is_json and "url" in request.json:
        image_url = request.json["url"]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    elif "file" in request.files:
        file = request.files["file"]
        image = Image.open(file.stream)
        image_url = None
    else:
        return jsonify({"error": "�������� ���� 'url' (JSON) ���� 'file' (form-data)"}), 400

    # ��������� ��� ������ ��� Roboflow
    result = client.run_workflow(
        workspace_name="cfu-counter",  # ���� �� ���� ���
        workflow_id="custom-workflow-4",  # ���� �� ���� ���
        images={"image": image_url} if image_url else {"image": file},
        use_cache=True
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)