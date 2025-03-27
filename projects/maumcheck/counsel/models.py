from django.db import models

class CounselQuestion(models.Model):
    subject = models.CharField(max_length=200)
    name = models.CharField(max_length=10)
    age = models.CharField(max_length=10)
    address = models.CharField(max_length=10)
    content = models.TextField()

    emotion = models.TextField()
    situation = models.TextField()
    hashtag = models.CharField(max_length=10)
    
    create_date = models.DateTimeField()