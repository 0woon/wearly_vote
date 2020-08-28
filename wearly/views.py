from django.shortcuts import render
from wearly.models import User, Image
from random import randint
from django.http import HttpResponse
from django import forms
from .forms import UserForm


# Create your views here.

def index(req):
    print("페이지 열기")

    # image = Image()

    image_list = Image.objects.order_by("?")[:100]

    context = {
        "image_list": image_list
    }

    return render(req, "index.html", context=context)


def vote(req):

    print("저장할게요")
    form = UserForm(req.POST)
    a = form.save(commit=False)
    # print(a.name)
    # print(a.age)hero
    # print(a.gender)
    # print(a.imageScore1)
    a.save()

    return HttpResponse("감사합니다")