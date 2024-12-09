from flask import Flask, request, jsonify
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline
from src.logger import logging

predict_pipeline = PredictPipeline()

app = Flask(__name__)
Features_to_exclude = [
    "sustainability",
    "bms_additoinal_info",
    "vms_in_scope",
    "fire_graphics_required",
    "gss_room_coverage",
    "fire_curtain_type",
    "lifting_pump_applicable",
    "ah_netbox_inScope"
]



@app.route('/', methods=["GET"])
def home():
    return "Just testing!"

@app.route('/predict',methods = ["POST"])
def predict():
    try:
        input_data = request.get_json()
        logging.info("Ok requested the website for data")
        features = pd.DataFrame([input_data])
        #features  = features.drop(columns=Features_to_exclude, errors="ignore")
        features = features.fillna(0)
        predictions = predict_pipeline.predict(features)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify',methods = ["POST"])
def classify():
    try:
        input_data = request.get_json()
        logging.info("Ok requested the website for data")
        features = pd.DataFrame([input_data])
        #features  = features.drop(columns=Features_to_exclude, errors="ignore")
        features = features.fillna(0)
        classification = predict_pipeline.classify(features)
        return jsonify({"predictions": classification})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)