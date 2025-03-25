## 필수 패키지 설치

```
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
```


## 가상환경 생성 및 Flask 설치
```
python3 -m venv venv
source venv/bin/activate  # 가상환경 활성화
pip install flask torch torchvision
```


## 모델 API

```
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

#  PyTorch 모델 로드
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 10)  # 예제 모델

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))  # CPU에서 로드
model.eval()

# 이미지 변환 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 예측 API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = model(image.view(1, -1))  # 모델에 입력
        prediction = output.argmax(dim=1).item()  # 최댓값 인덱스 가져오기

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 외부 접속 가능하게 실행

```


## API 실행 
```
cd ~/emotion_situation
source /home/ubuntu/venv/bin/activate
nohup python emotion_api.py > emotion.log 2>&1 &
nohup python situation_api.py > situation.log 2>&1 &
```

## 실행상태 확인
```
ps aux | grep python
```

```
(venv) ubuntu@ip-172-26-7-105:~/emotion_situation$ ps aux | grep python
root         465  0.0  0.6  33120 13176 ?        Ss   Mar19   0:00 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers
root         581  0.0  0.4 110136  9472 ?        Ssl  Mar19   0:00 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal
ubuntu     19417  0.0  0.1   7008  2304 pts/1    S+   06:05   0:00 grep --color=auto python
[2]-  Exit 1                  nohup python situation_api.py > situation.log 2>&1
[3]+  Exit 1                  nohup python situation_api.py > situation.log 2>&1

```



## curl 테스트 

```
curl -X POST http://13.124.53.104:5001/predict_emotion -H "Content-Type: application/json" -d '{"text": "기분이 너무 좋아!"}'

curl -X POST http://13.124.53.104:5001/predict_emotion \
     -H "Content-Type: application/json" \
     -d '{"text": "기분이 너무 좋아!"}'




```


```


curl -X POST http://13.124.53.104:5002/predict_situation \
     -H "Content-Type: application/json" \
     -d '{"text": "출근하기 싫어 "}'



curl -X POST http://13.124.53.104:5002/predict_situation -H "Content-Type: application/json" -d '{"text": "출근하기 싫어 "}'

```


```
tail -f emotion.log  # 감정 모델 로그 확인
tail -f situation.log  # 상황 모델 로그 확인

```








# local


##




```
C:\venvs>python -m venv mysite  
C:\venvs>python -m venv maumcheck
C:\Users\kj\Desktop\myGit\django\venvs\maumcheck\Scripts>activate
```
```
(maumcheck) (base) C:\Users\kj\Desktop\myGit\django\venvs\maumcheck\Scripts>
```



```
pip install flask
```



```
set FLASK_APP=emotion
set FLASK_DEBUG=True
flask run > log/emotion.log 2>&1
```


```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

```
python -c "import torch; import numpy; print(torch.__version__, numpy.__version__)"

```



```
export FLASK_RUN_PORT=5001
export FLASK_APP=situation.py
flask run > log/situation.log 2>&1


python emotion.py > log/situation.log 2>&1
python situation.py > log/situation.log 2>&1

```