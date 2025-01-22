import os
import sys
import importlib
import django
from django.contrib.auth.hashers import make_password

# 현재 디렉토리 확인
print("Current working directory:", os.getcwd())

# 프로젝트 루트 디렉토리 강제로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print("Updated sys.path:", sys.path)

# Django 설정 파일 로드
os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
print("DJANGO_SETTINGS_MODULE:", os.environ.get("DJANGO_SETTINGS_MODULE"))

# Django 설정 모듈 확인
try:
    settings = importlib.import_module("config.settings")
    print("Settings module loaded successfully:", settings)
except ModuleNotFoundError as e:
    print("Error loading settings module:", e)
    sys.exit(1)

# Django 초기화
try:
    django.setup()
    print("Django initialized successfully.")
except Exception as e:
    print("Error during Django setup:", e)
    sys.exit(1)

print("="*20)
# 비밀번호 해싱
hashed_password = make_password("1111")
print(f"Hashed password: {hashed_password}")
print("="*40)