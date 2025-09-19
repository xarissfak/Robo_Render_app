import os
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

# Δημιουργία Flask app
app = Flask(__name__)

# Φορτώνει το API key από τα Environment Variables του Render
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Απλό endpoint για έλεγχο
@app.route("/")
def home():
    return "✅ Το app τρέχει σωστά στο Render!"

# Endpoint για να στείλεις εικόνα στο μοντέλο
@app.route("/predict", methods=["POST"])
def predict():
    """
    Περιμένει ένα JSON request με:
    {
        "image_url": "https://path.to/your/image.jpg",
        "model_id": "MODEL/1"   # π.χ. "bacteria-detection/1"
    }
    """
    data = request.get_json()

    if not data or "image_url" not in data or "model_id" not in data:
        return jsonify({"error": "Χρειάζονται τα πεδία image_url και model_id"}), 400

    image_url = data["image_url"]
    model_id = data["model_id"]

    try:
        result = client.infer(image_url, model_id=model_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Εκκίνηση server
if __name__ == "__main__":
    # To Render ανοίγει το port από το env var PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
