from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import AutoTokenizer

app = Flask(__name__)

# 모델 로드 (Situation)
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = torch.load("situation_model.pth", map_location=torch.device("cpu"))
model.eval()

def predict_situation(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        output = model(**tokens)
        prediction = output.logits.argmax(dim=1).item()
    return prediction

@app.route('/predict_situation', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    prediction = predict_situation(text)
    return jsonify({'prediction_situation': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
