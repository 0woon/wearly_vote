from django.shortcuts import render
from wearly.models import User, Image, wear
from random import randint
from django.http import HttpResponse
from django import forms
from .forms import UserForm
from django_pandas.io import read_frame
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import time
from collections import deque
from collections import Counter

import tensorflow as tf
from six import next
from sklearn import preprocessing
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix




# Create your views here.

# 데이터 가져와서 유저 정보, 점수 나누기
def make_userANDrate():
    credentials = "postgresql://jczdtzmaouemml:f3f55cc0c6bd25a42866864c9299f4ec79ff4ff890f6f69467e2b14dc0010074@ec2-3-215-207-12.compute-1.amazonaws.com:5432/de8i6u9p9i6vq8"
    dbdf = pd.read_sql("""select distinct * from wearly_user where age <= 70 order by idx """, con=credentials)
    item = pd.read_sql("""select * from wearly_wear""", con=credentials)
    df = dbdf.copy()
    df = df.drop_duplicates(['name', 'gender', 'age'] + df.columns.tolist()[4:-1], keep='first').reset_index(drop=False)
    df = df.drop(['index', 'name', 'idx', 'time'], axis=1, inplace=False)

    print('데이터부르기 완료')

    v = [data_row.values.tolist()[2:] for index, data_row in df.iterrows()]
    vv = [v[i][j] for i in range(len(v)) for j in range(len(v[i]))]

    user_rt = pd.DataFrame(index=range(0, len(df) * 100), columns=['user', 'image_file_name', 'rate'])
    user_rt['user'] = sorted([i for i in range(0, len(df)) for j in range(0, 100)])
    user_rt['image_file_name'] = vv

    for i in range(len(user_rt)):
        user_rt['rate'][i] = int(user_rt['image_file_name'][i][-1])
        user_rt['image_file_name'][i] = str(user_rt['image_file_name'][i][:-1])

    user_rt = user_rt.sample(frac=1, random_state=200).reset_index(drop=True)
    user_rt = user_rt.merge(item[['image_id', 'image_file_name']], on='image_file_name')
    print('user_rt 생성 완료')

    user_rt.loc[user_rt['rate'] == 3, 'rate'] = 2
    user_rt.loc[user_rt['rate'] == 5, 'rate'] = 3
    print('숫자바꾸기 완료')
    PERC = 0.9
    rows = len(user_rt)
    split_index = int(rows * PERC)
    df_train = user_rt[0:split_index]
    df_test = user_rt[split_index:].reset_index(drop=True)
    print('split 완료')
    user_mt = df[['age', 'gender']]
    user_mt = user_mt.reset_index()
    user_mt['user'] = user_mt['index']
    user_mt = user_mt[['user', 'age', 'gender']]

    # get one-hot encoding (gender)
    user_mt = pd.get_dummies(user_mt, columns=["age", "gender"])

    return user_mt, df_train, df_test, item, user_rt


class ShuffleIterator(object):

    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))  # 0과 len사이의  batch_size 크기의 랜덤 정수 생성
        out = self.inputs[ids, :]  # 뭐임?
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len),
                                            np.ceil(self.len / batch_size))  # len 만큼의 array를 len/batch size의 올림만큼 분할
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]


def inferenceDense(MFSIZE, phase, user_batch, item_batch, idx_user, idx_item, user_num, item_num, UReg=0.05, IReg=0.1,
                   UW=0.05, IW=0.02):
    with tf.device('/cpu:0'):
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch,
                                            name="embedding_user")  # idx_user에서 user_batch의 index값을 뽑음
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")  # w_item 이 들어감

        ul1mf = tf.layers.dense(inputs=user_batch, units=MFSIZE, activation=tf.nn.crelu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        print(ul1mf.shape)
        il1mf = tf.layers.dense(inputs=item_batch, units=MFSIZE, activation=tf.nn.crelu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        print(il1mf.shape)
        InferInputMF = tf.multiply(ul1mf, il1mf)
        print(InferInputMF.shape)

        infer = tf.reduce_sum(InferInputMF, 1, name="inference")  # reduce_sum은 모든 차원제거하고 원소합

        regularizer = tf.add(UW * tf.nn.l2_loss(ul1mf), IW * tf.nn.l2_loss(il1mf), name="regularizer")  # l2 regularize
    return infer, regularizer, ul1mf, il1mf


def optimization(infer, regularizer, rate_batch, learning_rate=0.0005, reg=0.1):
    with tf.device('/cpu:0'):
        global_step = tf.train.get_global_step()  # 훈련 중단시 체크포인트
        assert global_step is not None
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))  # infer - rate_batch?
        cost = tf.add(cost_l2, regularizer)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op


def clip(x):
    return np.clip(x, 1.0, 3.0)  # 벗어나는 값들 위치시키기


def prediction_matrix(user_dict, item_dict):
    pred = np.zeros((len(user_dict), len(item_dict)))
    for i in range(len(user_dict)):
        for j in range(len(item_dict)):
            pred[i][j] += np.dot(user_dict[i], item_dict[j])
    return pred


def recommender_for_user(users_items_matrix_df, user_id, interact_matrix, df_content, topn=6):
    '''
    Recommender Games for UserWarning
    '''
    pred_scores = interact_matrix.loc[user_id].values

    df_scores = pd.DataFrame({'image_id': list(users_items_matrix_df.columns),
                              'score': pred_scores})

    df_rec = df_scores.set_index('image_id') \
        .join(df_content.set_index('image_id')) \
        .sort_values('score', ascending=False) \
        .head(topn)[['score', 'image_file_name', 'hashtag_crawl']]

    return df_rec[df_rec.score > 0]


def GraphRec(Mf, Epoch, NORM):
    tf.reset_default_graph()
    usrDat, df_train, df_test, itmDat, user_rt = make_userANDrate()

    USER_NUM = len(usrDat);
    ITEM_NUM = user_rt.image_id.nunique()
    BATCH_SIZE = 1000
    DEVICE = "/cpu:0"
    # With Graph Features
    MFSIZE = Mf
    UW = 0.05
    IW = 0.02
    LR = 0.00003
    EPOCH_MAX = Epoch

    UsrDat = usrDat.drop(['user'], axis=1, inplace=False)
    ItmDat = itmDat.drop(['idx', 'image_id', 'post_id', 'image_file_name', 'hashtag_crawl', 'account_name'], axis=1,
                         inplace=False)

    if (NORM):
        ItmDat['like_num'] = np.log10(ItmDat['like_num'].values + 1)
        ItmDat['comment_num'] = np.log10(ItmDat['comment_num'].values + 1)
    else:
        ItmDat = ItmDat.drop(['comment_num', 'like_num'], axis=1, inplace=False)

    UsrDat = UsrDat.values
    ItmDat = ItmDat.values

    AdjacencyUsers = np.zeros((USER_NUM, ITEM_NUM), dtype=np.float32)  # N x M shape의 zero matrix 생성 (Adjacency)
    DegreeUsers = np.zeros((USER_NUM, 1), dtype=np.float32)  # N x 1  shape의 zero vactor 생성 (Degree)

    AdjacencyItems = np.zeros((ITEM_NUM, USER_NUM), dtype=np.float32)  # M x N shape의 zero matrix 생성
    DegreeItems = np.zeros((ITEM_NUM, 1), dtype=np.float32)  # M X 1 shape의 zero vactor 생성
    for index, row in df_train.iterrows():
        userid = int(row['user'])  # row돌면서 'user'와 'item' column의 값 저장
        itemid = int(row['image_id'])
        AdjacencyUsers[userid][itemid] = row['rate'] / 3.0  # train set의 rating / max 값을 numpy matrix에 저장
        AdjacencyItems[itemid][userid] = row['rate'] / 3.0  # 동일, transpose matrix에
        DegreeUsers[userid][0] += 1
        DegreeItems[itemid][0] += 1

    DUserMax = np.amax(DegreeUsers)  # max값
    DItemMax = np.amax(DegreeItems)
    DegreeUsers = np.true_divide(DegreeUsers, DUserMax)  # DegreeUsers의 array들 전부를 Max값으로 나누기
    DegreeItems = np.true_divide(DegreeItems, DItemMax)

    AdjacencyUsers = np.asarray(AdjacencyUsers, dtype=np.float32)  # 정규화된 rating이 적힌 matrix를 array로
    AdjacencyItems = np.asarray(AdjacencyItems, dtype=np.float32)

    UserFeatures = np.concatenate((np.identity(USER_NUM, dtype=np.bool_), AdjacencyUsers, DegreeUsers),
                                  axis=1)  # np.identity concat
    print(UserFeatures.shape)
    ItemFeatures = np.concatenate((np.identity(ITEM_NUM, dtype=np.bool_), AdjacencyItems, DegreeItems), axis=1)

    UserFeatures = np.concatenate((UserFeatures, UsrDat), axis=1)

    ItemFeatures = np.concatenate((ItemFeatures, ItmDat), axis=1)

    UserFeaturesLength = UserFeatures.shape[1]
    ItemFeaturesLength = ItemFeatures.shape[1]

    print(UserFeatures.shape)
    print(ItemFeatures.shape)

    samples_per_batch = len(df_train) // BATCH_SIZE  # 90000 / 1000 = 90

    iter_train = ShuffleIterator([df_train["user"], df_train["image_id"], df_train["rate"]],
                                 batch_size=BATCH_SIZE)  # 1000 X 90 이나옴

    iter_test = OneEpochIterator([df_test["user"], df_test["image_id"], df_test["rate"]], batch_size=10000)  # 10000개?

    # tensor 값을 할당할 placeholder 생성
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")  # dtype,shape default
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')

    # tensor matrix생성
    w_user = tf.constant(UserFeatures, name="userids", shape=[USER_NUM, UserFeatures.shape[1]],
                         dtype=tf.float64)  # 943x2710을 constant
    w_item = tf.constant(ItemFeatures, name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]],
                         dtype=tf.float64)  # 1682x2646

    infer, regularizer, p, s = inferenceDense(MFSIZE, phase, user_batch, item_batch, w_user, w_item, user_num=USER_NUM,
                                              item_num=ITEM_NUM)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=LR, reg=0.09)

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    finalerror = -1
    # train_ls = []
    # test_ls= []
    p_ls = []
    s_ls = []

    p_dict = dict()
    s_dict = dict()

    total_df = user_rt.copy()
    iter_final = OneEpochIterator([total_df["user"], total_df["image_id"], total_df["rate"]],
                                  batch_size=len(total_df))  # 10000개?

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):  # 10 X 90
            # users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = next(iter_train)
            _, pred_batch, p_mat, s_mat = sess.run([train_op, infer, p, s], feed_dict={user_batch: users,
                                                                                       item_batch: items,
                                                                                       rate_batch: rates,
                                                                                       phase: True})
            pred_batch = clip(pred_batch)
            # train_ls.append(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:  # 1 Epoch 일때마다 / batch가 90단위마다
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                degreelist = list()
                predlist = list()
                for users, items, rates in iter_test:  # test의 pred_batch
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,
                                                            phase: False})

                    pred_batch = clip(pred_batch)
                    # test_ls.append(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                finalerror = test_err
                print("{:3d},{:f},{:f},{:f}(s)".format(i // samples_per_batch, train_err, test_err, end - start))
                start = end

        for users, items, rates in iter_final:  # test의 pred_batch
            pred_batch, p_mat, s_mat = sess.run([infer, p, s], feed_dict={user_batch: users,
                                                                          item_batch: items,
                                                                          phase: False})

            p_ls.append(p_mat)
            s_ls.append(s_mat)

            concat_p = np.vstack(p_ls)
            concat_s = np.vstack(s_ls)

            user_arr = total_df.user.values
            for idx, user in enumerate(user_arr):
                if user not in p_dict:
                    p_dict[user] = concat_p[idx]

            item_arr = total_df.image_id.values
            for idx, item in enumerate(item_arr):
                if item not in s_dict:
                    s_dict[item] = concat_s[idx]

    pred = prediction_matrix(p_dict, s_dict)

    # recommendation
    users_items_matrix_df = user_rt.pivot(index='user',
                                          columns='image_id',
                                          values='rate').fillna(0)

    new_users_items_matrix_df = pd.DataFrame(pred,
                                             columns=users_items_matrix_df.columns,
                                             index=users_items_matrix_df.index)

    recom = recommender_for_user(users_items_matrix_df, user_id=usrDat['user'].values.tolist()[-1],
                                 interact_matrix=new_users_items_matrix_df,
                                 df_content=itmDat)

    r = users_items_matrix_df.values.astype(np.float64)
    r[r == 0] = 'nan'
    RMSE = np.sqrt(np.nansum((r - pred) ** 2 / np.isfinite(r).sum()))
    print('Final Rmse :', RMSE)

    return recom


def get_style_str(df, rate_df, k):  # df : DB dataframe , rate_df , frequency

    df_ = df.loc[:, ['image_id', 'image_file_name']]
    hashtag_df = df.iloc[:, 7:57]
    style_df = df.iloc[:, 57:]

    idx_ls = []
    file_ls = [x for x in imgs_df.image_file_name]
    for i in file_ls:
        for idx, j in enumerate(df_.image_file_name):
            if i == j:
                idx_ls.append(idx)

    style_ls = []
    style_rt = style_df.iloc[idx_ls, :]
    for i in idx_ls:
        for j in style_rt.columns:
            ls_ = [x for x in j if style_rt.loc[i, j] == 1]
            if style_rt.loc[i, j] == 1:
                style_ls.append(j)

    return Counter(style_ls).most_common(k)


def get_hashtag_str(df, rate_df, k):  # df : DB dataframe , rate_df , frequency

    df_ = df.loc[:, ['image_id', 'image_file_name']]
    hashtag_df = df.iloc[:, 7:57]
    style_df = df.iloc[:, 57:]

    idx_ls = []
    file_ls = [x for x in imgs_df.image_file_name]
    for i in file_ls:
        for idx, j in enumerate(df_.image_file_name):
            if i == j:
                idx_ls.append(idx)

    hashtag_ls = []
    hashtag_rt = hashtag_df.iloc[idx_ls, :]
    for i in idx_ls:
        for j in hashtag_rt.columns:
            ls_ = [x for x in j if hashtag_rt.loc[i, j] == 1]
            if hashtag_rt.loc[i, j] == 1:
                hashtag_ls.append(j)

    hash_common = Counter(hashtag_ls).most_common(k)
    hash_ = [i[0] for i in hash_common]

    return hash_


# ========================================================================================
# 인덱스 페이지 열기 (유저 정보, 이미지 선택)
def index(req):
    print("페이지 열기")

    # image = Image()

    image_list = Image.objects.order_by("?")[:100]

    context = {
        "image_list": image_list
    }

    return render(req, "index.html", context=context)



# 이미지 선택 값 받은 후 recommend 페이지 보여주기
def vote(req):

    print("저장할게요")
    form = UserForm(req.POST)
    a = form.save(commit=False)
    # a.save()
    # recom = GraphRec(60,1, NORM=False)
    # print(recom)

    # dbData = wear.objects.all()
    # df = read_frame(dbData)


    #wear 테이블로 접속
    #데이터 값 -> dataframe화
    #가공
    #가공된 데이터를 바로 context로 보여줄수 있나?
    #아니면 가공된 데이터를 Model로 또 만들어서 넣고 출력해야하나?

    ################## user = User.objects.all() = select * from weary_user
                        # userDataframe = user
    # qs = wear.objects.all()
    # print(qs)
    # df = read_frame(qs)
    # print(df)

    # images = []
    #
    # for image in recom:
    #     images.append(recom[image].image_file_name)

    # recom.image_file_name.values.tolist()
    data = pd.read_csv("C:\\Users\\young\\Desktop\\wear_item.csv", index_col=0)
    # print(data.image_file_name.values.tolist()[:10])
    # data = wear.objects.all()[:10]
    # images = []
    # for image in data:
    #     images.append(image.image_file_name)
    # print(images)
    context = {
        "user" :User.objects.order_by('-pk')[0],
        "recInstas" : data.account_name.values.tolist()[:3],
        "recImages" : data.image_file_name.values.tolist()[:6],
        # "hashtags" data.image_file_name.values.tolist()[:6]:
    }

    # return HttpResponse("감사합니다")
    return render(req, "recommend.html", context=context)

