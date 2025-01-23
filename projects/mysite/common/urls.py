from django.urls import path
from django.contrib.auth import views as auth_views
from . import views


app_name = 'common'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='common/login.html'), name='login'),
    #  django.contrib.auth 앱의 LoginView를 사용하도록 설정 
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup, name='signup'),
]