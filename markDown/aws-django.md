## 시스템 패키지 업데이트 및 Python 설치 

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y


## Django 프로젝트 환경 설정


mkdir venvs


python3 -m venv maumcheck

source venvs/maumcheck/bin/activate

pip install django

프로젝트로 이동 후 

django-admin startproject config .




## Django 앱 생성 및 인덱스 페이지 만들기








python3 manage.py runserver 0.0.0.0:8000




```
sudo apt install authbind
sudo touch /etc/authbind/byport/80
sudo chown ubuntu /etc/authbind/byport/80
sudo chmod 755 /etc/authbind/byport/80

```
```
authbind --deep python3 manage.py runserver 0.0.0.0:80
```

```
nohup authbind --deep python3 manage.py runserver 0.0.0.0:80 &

```