from flask import Flask
from flask_cors import CORS


import re
import torch
import torch.nn.functional as F
from flask import request, jsonify


import logging
import os



app = Flask(__name__)


# # 코드 내에 명시적 로깅
app.logger.info("emotion request!")


# # 🔸 로그 폴더 없으면 생성
# if not os.path.exists("log"):
#     os.makedirs("log")

# # 🔸 로그 포맷 설정
# log_formatter = logging.Formatter(
#     '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
# )

# # ✅ 1. 파일 핸들러
# file_handler = logging.FileHandler("log/emotion.log", encoding='utf-8')
# file_handler.setFormatter(log_formatter)
# file_handler.setLevel(logging.INFO)

# # ✅ 2. 콘솔 핸들러
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# console_handler.setLevel(logging.INFO)

# # 🔸 Flask 기본 로거에 핸들러 등록
# app.logger.setLevel(logging.INFO)
# app.logger.addHandler(file_handler)
# app.logger.addHandler(console_handler)


CORS(app)  # 전체 도메인 허용
#CORS(app, resources={r"/emotion": {"origins": "http://vb901217.dothome.co.kr"}})


@app.route('/')
def hello():
    return 'hello'

import torch
import torch.nn as nn
from transformers import AutoTokenizer, ElectraModel


# 감정 분류 모델 정의
class KoELECTRAEmotion(nn.Module):
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
        category_logits = self.emotion_classifier(x)

        return category_logits


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_name = "monologg/koelectra-base-discriminator"  # ✅ KoELECTRA 적용
model = KoELECTRAEmotion(model_name, category=12)  # 모델 인스턴스 생성
model.load_state_dict(torch.load("model/emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# emotion_mapping = {
#     0: ('EM01', '기쁨'), 1: ('EM02', '슬픔'), 2: ('EM03', '분노'), 3: ('EM04', '혐오'),
#     4: ('EM05', '불안'), 5: ('EM06', '공포'), 6: ('EM07', '놀람'), 7: ('EM08', '수치심'),
#     8: ('EM09', '절망감'), 9: ('EM10', '죄책감'), 10: ('EM11', '우울/외로움'), 11: ('EM12', '기타')
# }


emotion_mapping = {
    0: "기쁨", 
    1: "슬픔", 
    2: "분노", 
    3: "불안",
    4: "죄책감", 
    5: "우울", 
    6: "당황", 
    7: "혐오",
    8: "놀람", 
    9: "외로움", 
    10: "무기력", 
    11: "기타"
}

situation_mapping = {
    0: '가족 및 대인관계', 
    1: '학업 및 진로',
    2: '직장 및 업무 스트레스',
    3: '정신 건강 문제', 
    4: '학교폭력 및 학대',
    5: '중독 문제',
    6: '경제적 어려움',
    7: '신체 건강 문제', 
    8: '죽음과 상실',
    9: '인지 기능 관련 어려움', 
    10: '민감한 피해 경험', 
    11: '연애/결혼/출산'
}




@app.route("/emotion", methods=["POST"])
def predict_emotion():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({'error': 'No text provided'}), 400

        # 텍스트 정리
        text = re.sub(r"<.*?>", "", data["text"]).strip()
        text = preprocess_text(text)
        app.logger.debug(f"[DEBUG] text: {text}")
        if len(text) < 5:
            return jsonify({
                "input_text": text,
                "top3_predictions": [{"emotion": "분석불가", "probability": 1.0}],
                "keywords": []
            })

        # Tokenization
        encoding = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))

        # Chunking
        def pad_and_chunk(input_ids, attention_mask, token_type_ids, start, max_len):
            end = min(start + max_len, len(input_ids))
            chunk_input_ids = input_ids[start:end].unsqueeze(0)
            chunk_attention_mask = attention_mask[start:end].unsqueeze(0)

            if token_type_ids.dim() == 1:
                chunk_token_type_ids = token_type_ids[start:end].unsqueeze(0)
            else:
                chunk_token_type_ids = token_type_ids[:, start:end]

            pad_len = max_len - chunk_input_ids.size(1)
            if pad_len > 0:
                pad = torch.zeros((1, pad_len), dtype=torch.long)
                chunk_input_ids = torch.cat([chunk_input_ids, pad], dim=1)
                chunk_attention_mask = torch.cat([chunk_attention_mask, pad], dim=1)
                chunk_token_type_ids = torch.cat([chunk_token_type_ids, pad], dim=1)

            return {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "token_type_ids": chunk_token_type_ids
            }

        MAX_LEN, STRIDE = 512, 256
        chunks = [
            pad_and_chunk(input_ids, attention_mask, token_type_ids, start, MAX_LEN)
            for start in range(0, len(input_ids), STRIDE)
        ]

        if not chunks:
            return jsonify({
                "input_text": text,
                "top3_predictions": [{"emotion": "기타", "probability": 1.0}],
                "keywords": []
            })

        total_probs = torch.zeros(len(emotion_mapping)).to(device)
        keywords_all = []

        for chunk in chunks:
            tokens = {k: v.to(device) for k, v in chunk.items()}

            with torch.no_grad():
                logits = model(**tokens)
                probs = F.softmax(logits, dim=1)[0]
                total_probs += probs

                # 🔍 중심 키워드 추출
                try:
                    attn = model.last_attention[-1][0]
                    cls_att = attn[0]
                    token_strs = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

                    if cls_att.shape[0] == len(token_strs):
                        token_scores = cls_att[1:].mean(dim=0)
                        keyword_scores = list(zip(token_strs[1:], token_scores.tolist()))
                        keyword_scores.sort(key=lambda x: x[1], reverse=True)

                        merged = merge_wordpieces(keyword_scores)
                        merged.sort(key=lambda x: x[1], reverse=True)

                        app.logger.debug(f"[DEBUG] merged tokens: {merged[:10]}")

                        if merged:
                            top_tokens = [token for token, _ in merged[:5]]
                            for core_token in top_tokens:
                                context = extract_context_window(text, core_token, window_size=30)
                                app.logger.debug(f"[DEBUG] context for '{core_token}': {context}")
                                context_keywords = extract_keywords_from_text(context)
                                keywords_all.extend(context_keywords)
                except Exception as ke:
                    app.logger.warning(f"[키워드 추출 오류] {str(ke)}")

        avg_probs = total_probs / len(chunks)
        top3_probs, top3_indices = torch.topk(avg_probs, k=3)

        top3_results = [
            {
                "emotion": emotion_mapping.get(idx.item(), f"분류불가({idx.item()})"),
                "probability": round(prob.item(), 4)
            }
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        # unique_keywords = list(dict.fromkeys(keywords_all))[:20]
        # if not unique_keywords:
        #     unique_keywords = ["(키워드 없음)"]

        unique_keywords = filter_final_keywords(keywords_all, top_k=10)
        if not unique_keywords:
            unique_keywords = ["(키워드 없음)"]

        return jsonify({
            "input_text": text,
            "top3_predictions": top3_results,
            "keywords": unique_keywords
        })

    except Exception as e:
        app.logger.error(f"[ERROR] 감정 예측 실패: {str(e)}", exc_info=True)
        return jsonify({'error': f"서버 오류: {str(e)}"}), 500
















def preprocess_text(text):
    important_chars = "!?,~ㅠㅋ^…"
    text = ''.join([char if char in important_chars or char.isalnum() else ' ' for char in text])
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'(ㅋ)\1+', r'\1\1', text)
    text = re.sub(r'(ㅠ)\1+', r'\1\1', text)
    return text.strip()



# WordPiece 토큰을 결합하여 원래 단어 형태로 복원
def merge_wordpieces(token_scores):
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





from konlpy.tag import Okt
from kiwipiepy import Kiwi
from collections import OrderedDict

okt = Okt()
kiwi = Kiwi()



#  주어진 토큰이 포함된 문장을 원문에서 찾아 반환

def find_sentence_with_token(token, text):
    token_clean = token.replace("##", "")
    sentences = re.split(r'[.!?\n]', text)  # 문장 분리

    for sent in sentences:
        if token_clean in sent:
            return sent.strip()
    return token_clean  # fallback: 토큰 자체



# 해당 문장에서 Okt와 Kiwi 형태소 분석기를 활용하여 명사, 동사, 형용사 추출
def extract_keywords_from_text(text):
    keywords = set()
    try:
        okt_tokens = okt.pos(text, stem=True)
        keywords.update([word for word, tag in okt_tokens if tag in ["Noun", "Verb", "Adjective"] and len(word) >= 2])
    except Exception as e:
        app.logger.warning(f"[Okt 분석 오류] {str(e)}")

    try:
        kiwi_tokens = kiwi.analyze(text)
        if kiwi_tokens:
            for token in kiwi_tokens[0][0]:
                if token.tag in ["NNG", "NNP", "VV", "VA"] and len(token.form) >= 2:
                    keywords.add(token.form)
    except Exception as e:
        app.logger.warning(f"[Kiwi 분석 오류] {str(e)}")

    return list(keywords)



def extract_context_window(text, token, window_size=30):
    token_clean = token.replace("##", "")
    idx = text.find(token_clean)
    if idx == -1:
        return token_clean
    start = max(0, idx - window_size)
    end = min(len(text), idx + len(token_clean) + window_size)
    return text[start:end]


def filter_final_keywords(keywords, top_k=10):
    allowed_tags = {"Noun", "Verb", "Adjective"}
    seen = set()
    filtered = []

    for kw in keywords:
        if len(kw) < 2:  # 너무 짧은 단어 제외
            continue
        if kw in seen:
            continue
        seen.add(kw)
        filtered.append(kw)
        if len(filtered) >= top_k:
            break

    return filtered



# from collections import OrderedDict
# from kiwipiepy import Kiwi
# from konlpy.tag import Okt

# kiwi = Kiwi()
# okt = Okt()

# def find_sentence_with_token(token, text):
#     for sentence in text.split('\n'):
#         if token.replace("##", "") in sentence:
#             return sentence
#     return text

# def extract_keywords_from_top_tokens(text, top_tokens, top_k=3):
#     keywords = []
#     for token in top_tokens:
#         sentence = find_sentence_with_token(token, text)
#         morphs = extract_keywords_from_text(sentence)
#         keywords.extend(morphs)
#     unique_keywords = list(OrderedDict.fromkeys(keywords))
#     return unique_keywords[:top_k]

# def extract_keywords_from_text(text):
#     tokens = okt.pos(text, stem=True)
#     return [word for word, tag in tokens if tag in ["Noun", "Verb", "Adjective"] and len(word) >= 2]


# def merge_wordpieces(token_scores):
#     merged_tokens = []
#     current_word = ""
#     current_score = 0.0
#     count = 0

#     for token, score in token_scores:
#         if token.startswith("##"):
#             current_word += token[2:]
#             current_score += score
#             count += 1
#         else:
#             if current_word:
#                 merged_tokens.append((current_word, current_score / count))
#             current_word = token
#             current_score = score
#             count = 1

#     # 마지막 단어 처리
#     if current_word:
#         merged_tokens.append((current_word, current_score / count))

#     # [PAD], [SEP] 같은 무의미 토큰 제거
#     merged_tokens = [(w, s) for w, s in merged_tokens if w not in ['[PAD]', '[SEP]', '[CLS]']]

#     return merged_tokens




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


# if __name__ == "__main__":
#     debug_mode = os.environ.get("DEBUG_MODE", "True") == "True"
#     app.run(
#         host="0.0.0.0" if not debug_mode else "127.0.0.1",
#         port=5001,
#         debug=debug_mode,
#         use_reloader=debug_mode
#     )
