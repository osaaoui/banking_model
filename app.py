from flask import Flask, render_template, request, jsonify
from utils import model_predict
#app = Flask(__name__)
import os
import numpy as np

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('content')
    prediction_category, prediction_issue = model_predict(email)
    return render_template("index.html", prediction_category=prediction_category,prediction_issue=prediction_issue, email=email)

# Create an API endpoint
@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    prediction_category, prediction_issue = model_predict(email)
    #result= model_predict(email)
    return jsonify({'prediction_category': prediction_category, 'Prediction_issue': prediction_issue})  # Return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
