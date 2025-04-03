"""
🔸 통합 Flask 서버
- /emotion → 감정 예측
- /situation → 상황 분류
"""

# ✅ 공통 라이브러리 및 설정
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, ElectraModel

import re
import logging
import os

from konlpy.tag import Okt
from kiwipiepy import Kiwi

# ✅ Flask 앱 정의
app = Flask(__name__)


#

allowed_origins = [
    "http://vb901217.dothome.co.kr",
    "http://maumcheck.site"
]

#CORS(app)

CORS(app, resources={

    r"/emotion": {"origins": allowed_origins},
    r"/situation": {"origins": allowed_origins},

})

# ✅ 형태소 분석기 초기화
okt = Okt()
kiwi = Kiwi()

# ✅ 디바이스 설정
device = torch.device("cpu")

# ✅ KoELECTRA 공통 설정
model_name = "monologg/koelectra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --------------------------------------------
# 🔹 감정 분류 모델 정의 및 로드
# --------------------------------------------
class KoELECTRAEmotion(nn.Module):
    """KoELECTRA 기반 감정 분류 모델"""
    def __init__(self, model_name, category=12, dropout_prob=0.1):
        super(KoELECTRAEmotion, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.emotion_classifier = nn.Linear(256, category)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=True
        )
        self.last_attention = outputs.attentions
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.fc(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        return self.emotion_classifier(x)

emotion_model = KoELECTRAEmotion(model_name, category=12)
emotion_model.load_state_dict(torch.load("model/emotion_model.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()

emotion_mapping = {
    0: "기쁨", 1: "슬픔", 2: "분노", 3: "불안",
    4: "죄책감", 5: "우울", 6: "당황", 7: "혐오",
    8: "놀람", 9: "외로움", 10: "무기력", 11: "기타"
}

# --------------------------------------------
# 🔹 상황 분류 모델 정의 및 로드
# --------------------------------------------
class KoELECTRASituation(nn.Module):
    """KoELECTRA 기반 상황 분류 모델"""
    def __init__(self, model_name, category=12, dropout_prob=0.1):
        super(KoELECTRASituation, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.situation_classifier = nn.Linear(256, category)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=True
        )
        self.last_attention = outputs.attentions
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.fc(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        return self.situation_classifier(x)

situation_model = KoELECTRASituation(model_name, category=12)
situation_model.load_state_dict(torch.load("model/situation_model.pth", map_location=device))
situation_model.to(device)
situation_model.eval()

situation_mapping = {
    0: '가족 및 대인관계', 1: '학업 및 진로', 2: '직장 및 업무 스트레스',
    3: '정신 건강 문제', 4: '학교폭력 및 학대', 5: '중독 문제',
    6: '경제적 어려움', 7: '신체 건강 문제', 8: '죽음과 상실',
    9: '인지 기능 관련 어려움', 10: '민감한 피해 경험', 11: '연애/결혼/출산'
}

# --------------------------------------------
# 🔹 공통 함수 정의 (전처리, 키워드 추출 등)
# --------------------------------------------
def preprocess_text(text):
    """특수문자 제거 및 반복 문자 정리"""
    important_chars = "!?,~ㅠㅋ^…"
    text = ''.join([char if char in important_chars or char.isalnum() else ' ' for char in text])
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'(ㅋ)\1+', r'\1\1', text)
    text = re.sub(r'(ㅠ)\1+', r'\1\1', text)
    return text.strip()

def merge_wordpieces(token_scores):
    """WordPiece 토큰을 결합하여 원래 단어 복원"""
    merged_tokens = []
    current_word = ""
    current_score = 0.0
    count = 0
    for token, score in token_scores:
        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
            count += 1
        else:
            if current_word:
                merged_tokens.append((current_word, current_score / count))
            current_word = token
            current_score = score
            count = 1
    if current_word:
        merged_tokens.append((current_word, current_score / count))
    return merged_tokens

def extract_context_window(text, token, window_size=30):
    """해당 토큰이 포함된 주변 문맥 추출"""
    token_clean = token.replace("##", "")
    idx = text.find(token_clean)
    if idx == -1:
        return token_clean
    start = max(0, idx - window_size)
    end = min(len(text), idx + len(token_clean) + window_size)
    return text[start:end]

def extract_keywords_from_text(text):
    """Okt 및 Kiwi로 명사/동사/형용사 추출"""
    keywords = set()
    try:
        okt_tokens = okt.pos(text, stem=True)
        keywords.update([word for word, tag in okt_tokens if tag in ["Noun", "Verb", "Adjective"] and len(word) >= 2])
    except:
        pass
    try:
        kiwi_tokens = kiwi.analyze(text)
        for token in kiwi_tokens[0][0]:
            if token.tag in ["NNG", "NNP", "VV", "VA"] and len(token.form) >= 2:
                keywords.add(token.form)
    except:
        pass
    return list(keywords)

def filter_final_keywords(keywords, top_k=10):
    """최종 키워드 필터링 (중복 제거, 길이 제한 등)"""
    seen = set()
    filtered = []
    for kw in keywords:
        if len(kw) < 2 or kw in seen:
            continue
        seen.add(kw)
        filtered.append(kw)
        if len(filtered) >= top_k:
            break
    return filtered





@app.route("/")
def hello():
    return "hello, myModel"


# --------------------------------------------
# 🔹 감정 예측 라우트
# --------------------------------------------
@app.route("/emotion", methods=["POST"])
def predict_emotion():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = preprocess_text(data["text"])
    encoding = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))

    MAX_LEN, STRIDE = 512, 256
    chunks = [
        {
            "input_ids": input_ids[start:start + MAX_LEN].unsqueeze(0),
            "attention_mask": attention_mask[start:start + MAX_LEN].unsqueeze(0),
            "token_type_ids": token_type_ids[start:start + MAX_LEN]
        }
        for start in range(0, len(input_ids), STRIDE)
    ]

    total_probs = torch.zeros(len(emotion_mapping)).to(device)
    keywords_all = []

    for chunk in chunks:
        tokens = {k: v.to(device) for k, v in chunk.items()}
        with torch.no_grad():
            logits = emotion_model(**tokens)
            probs = F.softmax(logits, dim=1)[0]
            total_probs += probs

            attn = emotion_model.last_attention[-1][0]
            cls_att = attn[0]
            token_strs = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
            if cls_att.shape[0] == len(token_strs):
                token_scores = list(zip(token_strs[1:], cls_att[1:].mean(dim=0).tolist()))
                merged = merge_wordpieces(token_scores)
                merged.sort(key=lambda x: x[1], reverse=True)
                for token, _ in merged[:5]:
                    ctx = extract_context_window(text, token)
                    keywords_all.extend(extract_keywords_from_text(ctx))

    avg_probs = total_probs / len(chunks)
    top3_probs, top3_indices = torch.topk(avg_probs, k=3)
    top3_results = [
        {"emotion": emotion_mapping[idx.item()], "probability": round(prob.item(), 4)}
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    return jsonify({
        "input_text": text,
        "top3_predictions": top3_results,
        "keywords": filter_final_keywords(keywords_all)
    })

# --------------------------------------------
# 🔹 상황 예측 라우트
# --------------------------------------------
@app.route("/situation", methods=["POST"])
def predict_situation():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = preprocess_text(data["text"])
    encoding = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))

    MAX_LEN, STRIDE = 512, 256
    chunks = [
        {
            "input_ids": input_ids[start:start + MAX_LEN].unsqueeze(0),
            "attention_mask": attention_mask[start:start + MAX_LEN].unsqueeze(0),
            "token_type_ids": token_type_ids[start:start + MAX_LEN]
        }
        for start in range(0, len(input_ids), STRIDE)
    ]

    total_probs = torch.zeros(len(situation_mapping)).to(device)
    keywords_all = []

    for chunk in chunks:
        tokens = {k: v.to(device) for k, v in chunk.items()}
        with torch.no_grad():
            logits = situation_model(**tokens)
            probs = F.softmax(logits, dim=1)[0]
            total_probs += probs

            attn = situation_model.last_attention[-1][0]
            cls_att = attn[0]
            token_strs = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
            if cls_att.shape[0] == len(token_strs):
                token_scores = list(zip(token_strs[1:], cls_att[1:].mean(dim=0).tolist()))
                merged = merge_wordpieces(token_scores)
                merged.sort(key=lambda x: x[1], reverse=True)
                for token, _ in merged[:5]:
                    ctx = extract_context_window(text, token)
                    keywords_all.extend(extract_keywords_from_text(ctx))

    avg_probs = total_probs / len(chunks)
    top3_probs, top3_indices = torch.topk(avg_probs, k=3)
    top3_results = [
        {"situation": situation_mapping[idx.item()], "probability": round(prob.item(), 4)}
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    return jsonify({
        "input_text": text,
        "top3_predictions": top3_results,
        "keywords": filter_final_keywords(keywords_all)
    })

# --------------------------------------------
# 🔹 서버 실행
# --------------------------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)