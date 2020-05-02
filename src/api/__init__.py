import logging
from pathlib import Path

import torch
from flask import Flask, jsonify, request

from model.regression import LinearRegressionModel

logging.basicConfig(level=logging.DEBUG)

model=torch.load(str(Path.cwd())+'/src/model/model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    number = request.args.get('number')
    x_data = torch.Tensor([[number]])
    result = model(x_data).item()
    return jsonify({'data': result})
