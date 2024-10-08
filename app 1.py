import numpy as np
import pickle
from flask import Flask, request, jsonify,send_file

app = Flask(__name__)

# Load model and LabelEncoder
with open('model_and_encoder.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        features[0, :2] = label_encoder.transform(features[0, :2])  # Transform team names
        prediction = model.predict(features)
        return jsonify({"predictions": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False)
