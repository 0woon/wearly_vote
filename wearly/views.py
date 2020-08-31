from django.shortcuts import render
from wearly.models import User, Image
from random import randint
from django.http import HttpResponse
from django import forms
from .forms import UserForm
# import pandas as pd
# import numpy as np


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
    # a.save()
    # print(a.name)
    # print(a.age)hero
    # print(a.gender)
    # print(a.imageScore1)


    ################## user = User.objects.all() = select * from weary_user
                        # userDataframe = user
    # user = User.objects.get(pk=1)
    # userDataFrame = user.to_dataframe()
    #
    # df = userDataFrame.copy()
    #
    # v = [data_row.values.tolist()[n:] for index, data_row in df.iterrows()]
    # vv = [v[i][j] for i in range(len(v)) for j in range(len(v[i]))]
    #
    # udf = pd.DataFrame(index=range(0,len(df)*100), columns=['user','image_file_name','rate'])
    # udf['user'] = sorted([i for i in range(0,len(df)) for j in range(0,100)])
    # udf['image_file_name'] = vv
    #
    # for i in range(len(udf)):
    #     udf['rate'][i] = int(udf['image_file_name'][i][-1])
    #     udf['image_file_name'][i] = str(udf['image_file_name'][i][:-13])
    #     print(udf)

    # print(user.imageScore3)
    # b = user.object_or(pk=1)
    # print(b)

    return HttpResponse("감사합니다")