
```
pip install flask-cors


from flask_cors import CORS



CORS(app)  # ✅ 전체 도메인 허용

# 또는 특정 도메인만 허용하고 싶을 경우:
# CORS(app, resources={r"/predict_emotion": {"origins": "http://vb901217.dothome.co.kr"}})

```