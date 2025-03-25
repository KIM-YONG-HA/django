import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

import json
import re



app = Flask(__name__)


CORS(app)  # 전체 도메인 허용

# 또는 특정 도메인만 허용하고 싶을 경우:
#CORS(app, resources={r"/predict_emotion": {"origins": "http://vb901217.dothome.co.kr"}})


import torch
import torch.nn as nn
from transformers import ElectraModel

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
        outputs = self.electra(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=True
            )


        self.last_attention = outputs.attentions  # ✅ 이걸로 attention 기반 키워드 추출 가능


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

model_name = ""
device = torch.device("cpu")
model_name = "monologg/koelectra-base-discriminator"  
model = KoELECTRAEmotion(model_name)  # 모델 인스턴스 생성
model.load_state_dict(torch.load("/home/ubuntu/emotion_situation/model/emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ✅ 감정 ID → 한글 라벨 매핑
emotion_mapping = {
    0: ('EM01', '기쁨'), 1: ('EM02', '슬픔'), 2: ('EM03', '분노'), 3: ('EM04', '혐오'),
    4: ('EM05', '불안'), 5: ('EM06', '공포'), 6: ('EM07', '놀람'), 7: ('EM08', '수치심'),
    8: ('EM09', '절망감'), 9: ('EM10', '죄책감'), 10: ('EM11', '우울/외로움'), 11: ('EM12', '기타')
}


# ✅ 감정 예측 API
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # 입력 텍스트 전처리 (HTML 태그 제거 등)
        text = re.sub(r"<.*?>", "", data['text'])

        # 토큰화 및 텐서로 변환
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # 감정 예측
        with torch.no_grad():
            logits = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens["token_type_ids"]
            )
            probs = F.softmax(logits, dim=1)[0]
            top3_probs, top3_indices = torch.topk(probs, k=3)

        # 결과 구성
        top3_results = [
            {
                "emotion": emotion_mapping.get(idx.item(), "알 수 없음"),
                "probability": round(prob.item(), 4)
            }
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        return jsonify({
            "input_text": text,
            "top3_predictions": top3_results
        })

    except Exception as e:
        print(f"[ERROR] 예외 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    
