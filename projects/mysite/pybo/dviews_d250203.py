#from django.http import HttpResponse
# from django.http import HttpResponseNotAllowed
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import Question, Answer
from .forms import QuestionForm, AnswerForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required

from django.contrib import messages



# def index(request):
#     return HttpResponse("hello django!")



# def index(request):
#     question_list = Question.objects.order_by('-create_date')
#     context = {'question_list': question_list}
#     return render(request, 'pybo/question_list.html', context)



def index(request):
    page = request.GET.get('page', 1) #  페이지
    question_list = Question.objects.order_by('-create_date')
    paginator = Paginator(question_list, 10) #  페이지당 10개
    page_obj = paginator.get_page(page)
    context = {'question_list': page_obj}
    return render(request, 'pybo/question_list.html', context)



def detail(request, question_id):
    #question = Question.objects.get(id=question_id)
    question = get_object_or_404(Question, pk=question_id)
    context = {'question': question}
    return render(request, 'pybo/question_detail.html', context)


# def answer_create(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     question.answer_set.create(content=request.POST.get('content'), create_date=timezone.now())
    
#     # import .modesl Question, Answer 추가 필요 
#     #answer = Answer(question=question, content=request.POST.get('content'), create_date=timezone.now())
#     #answer.save()
    
#     return redirect('pybo:detail', question_id=question_id)






@login_required(login_url='common:login')
def question_create(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            question.create_date = timezone.now()
            question.save()
            return redirect('pybo:index')
    else:
        form = QuestionForm()

    context = {'form':form}
    return render(request, 'pybo/question_form.html' , context)


# def answer_create(request, question_id):
#     """
#     pybo 답변 등록
#     """
#     question = get_object_or_404(Question, pk=question_id)
#     if request.method == 'POST':
#         form = AnswerForm(request.POST)
#         if form.is_valid():
#             answer = form.save(commit=False)
#             answer.author = request.user
#             answer.create_date = timezone.now()
#             answer.question = question
#             answer.save()
#             return redirect('pybo:detail', question_id=question.id)
#     else:
#         return HttpResponseNotAllowed('Only POST is possible.')
    
#     context = {'question': question, 'form': form}
#     return render(request, 'pybo/question_detail.html', context)



@login_required(login_url='common:login')
def answer_create(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user
            answer.create_date = timezone.now()
            answer.question = question
            answer.save()
            return redirect('pybo:detail', question_id=question_id)
    else:
        form = AnswerForm()

    context = {'question': question, 'form': form}
    return render(request, 'pybo/question_detail.html', context)



@login_required(login_url='common:login')
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        message.error(request, '수정 권한이 없습니다.')
        return redirect('pybo:detail', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()
            question.save()
            return redirect('pybo:detail', question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {'form': form}
    return render(request, 'pybo/question_form.html', context)

@login_required(login_url='common:login')
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id)    
    if request.user != question.author:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('pybo:detail', question_id=question.id)
    question.delete()
    return redirect('pybo:index')


@login_required(login_url='common:login')
def answer_modify(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('pybo:detail', question_id=answer.question.id)
    if request.method == 'POST':
        form = AnswerForm(request.POST, instance=answer)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.modify_date = timezone.now()
            answer.save()
            return redirect('pybo:detail', question_id=answer.question.id)
    else:
        form = AnswerForm(instance=answer)
    
    context = {'answer': answer, 'form': form}
    return render(request, 'pybo/answer_form.html', context)


@login_required(login_url='common:login')        
def answer_delete(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, '댓글 삭제 권한이 없습니다.')
    else:
        answer.delete()
    return redirect('pybo:detail', question_id=answer.question.id)
    
            





