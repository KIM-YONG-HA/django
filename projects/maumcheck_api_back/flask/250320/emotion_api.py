import torch
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

class KoELECTRAEmotion(nn.Module):
    def __init__(self, model_name, num_emotions=6, dropout_prob=0.1):
        super(KoELECTRAEmotion, self).__init__()

        # KoELECTRA 모델 로드
        self.electra = ElectraModel.from_pretrained(model_name)

        # Dropout 및 Fully Connected Layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()

        # 감정 분류 (6개)
        self.emotion_classifier = nn.Linear(256, num_emotions)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # KoELECTRA Forward
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # CLS 토큰 출력 (768차원)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 공통 레이어
        x = self.fc(cls_output)
        x = self.relu(x)
        x = self.dropout(x)

        # 감정 및 상황 예측
        emotion_logits = self.emotion_classifier(x)  # 감정 분류 (6개)

        return emotion_logits

# ✅ 모델 불러오기 (state_dict 로드 방식)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = KoELECTRASituation("monologg/koelectra-base-v3-discriminator")  # 모델 인스턴스 생성
model.load_state_dict(torch.load("/home/ubuntu/emotion_situation/model/emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")





@app.route('/predict_emotion', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        output = model(**tokens)
        prediction = output.argmax(dim=1).item()

    return jsonify({'prediction_emotion': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
