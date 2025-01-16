from django import forms
from pybo.models import Question, Answer


# class QuestionForm(forms.ModelForm):
#     class Meta:
#         #  사용할 모델
#         model = Question
#         #  QuestionForm에서 사용할 Question 모델의 속성
#         fields = ['subject', 'content']


# class QuestionForm(forms.ModelForm):
#     class Meta:
#         model = Question
#         fields = ['subject', 'content']
#         widgets = {
#             'subject': forms.TextInput(attrs={'class':'form-control'}),
#             'content': forms.Textarea(attrs={'class':'form-control', 'rows':10})
#         }
#         labels = {
#             'subject':'제목',
#             'content':'내용'
#         }


class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question 
        fields = ['subject', 'content']
        labels = {
            'subject':'제목',
            'content':'내용'
        }


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content':'답변내용'
        }




