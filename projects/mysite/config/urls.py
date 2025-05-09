"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
#from pybo import views
from pybo.views import base_views

urlpatterns = [
    path("admin/", admin.site.urls),
    #path("pybo/", base_views.index),
    #path('pybo/<int:question_id>/', views.detail)
    path('pybo/', include('pybo.urls')),
    path('common/', include('common.urls')), #  URL/common/으로 시작하는 url은 모두 common.urls 참조
    path('', base_views.index, name='index'),

]
