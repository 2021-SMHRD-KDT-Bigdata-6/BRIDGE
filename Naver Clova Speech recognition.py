# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 로직 하나로 통합

# 형태소 분류 함수
def lemmatize(word):
    morphtags = Komoran().pos(word)
    if morphtags[0][1] == 'NNG' or morphtags[0][1] == 'NNP':
        return morphtags[0][0]


# +
# 라이브러리 및 모델 불러오기

from flask import Flask ,render_template,request, redirect

from konlpy.tag import Kkma,Okt, Twitter, Komoran # 형태소 분석 라이브러리
from moviepy.editor import * # 영상을 오디오 파일로 변환
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.editor as mp
from pytube import YouTube # 유튜브 영상 다운로드 또는 불러오기
import pytube
import tqdm as tq

import kss # 텍스트 문장으로 바꾸는 라이브러리
import speech_recognition as sr # 오디오 파일 또는 음성을 텍스트로 변환
import pandas as pd
import numpy as np
# BOW = BAG of WORD : 단어가방, 단어모음, 단어사전
from sklearn.feature_extraction.text import CountVectorizer
# 위 도구는 빈도수 기반 벡터화 도구
import requests
import json
# -

yt.title


# +
# 형태소 분류 함수
def lemmatize(word):
    morphtags = Komoran().pos(word)
    if morphtags[0][1] == 'NNG' or morphtags[0][1] == 'NNP':
        return morphtags[0][0]
    
    
# naver CLOVA speech recognition API
class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/2049/3126b3ac1112b1ddf946124c4aad59679bbdd7b31c66ad2ab6f8955fab595d67'
    # Clova Speech secret key
    secret = '4127153cf8d74bc6aeb2042941aa77a1'

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response

app = Flask(__name__)

@app.route('/post',methods=['POST','GET'])
def result():
    if request.method == 'POST' :
        result = request.form
        print(result)
        yt = pytube.YouTube(result.get('link'))
        print(yt.title)
        
        # 유튜브 영상 다운로드 후 저장
        stream = yt.streams.all()[0]
        stream.download(output_path='test/data')
        
        # 영상 제목
        title = yt.title
        
        res = ClovaSpeechClient().req_upload(file='test/data/'+title+'.3gpp', completion='sync')
        #print(res.text)
        
        # 전체 텍스트를 json 타입 변수에 저장
        # 텍스트 추출
        text = json.loads(res.text)

        # kss 활용 텍스트 문장 화
        word_list = kss.split_sentences(text['text'])

        word_list=word_list[0:5]
        # 문장 끝 마침표 제거

        for i in range(len(word_list)):
            word_list[i]=word_list[i].replace('.','')
            
            
            # 명사만 가져오기 위한 삭제
        okt = Okt()
        headline = []
        stopwords = [ '의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','등','으로도']
        for sentence in word_list:
            temp = []
            # morphs() : 형태소 단위로 토큰화
            # stem = True : 형태소에서 어간을 추출
            temp = okt.morphs(sentence, stem = True)
            temp = [word for word in temp if not word in stopwords]
            headline.append(temp)

           # konlpy 트위터 이용 형태소 분류
        twitter = Twitter()
        sentences_tag_last=[]
        for word in headline:
            sentences_tag=[]
            for i in word :
                morph = twitter.pos(i)
                sentences_tag.append(morph)
            sentences_tag_last.append(sentences_tag)

        #  형태소 분류
        adj_list_last=[]
        for ko in sentences_tag_last:
            noun_adj_list=[]
            for i in ko:
                for word, tag in i:
                    if tag in ['Noun','Verb','Number','Adjective','Adverb','Alpha']:
                        noun_adj_list.append(word)
            adj_list_last.append(noun_adj_list)

        print(adj_list_last)
        # 형태소 분류
        for i in range(len(adj_list_last)):
            for j in range(len(adj_list_last[i])):
                if lemmatize(adj_list_last[i][j]) != None :
                        adj_list_last[i][j] = lemmatize(adj_list_last[i][j])

        arr_list = adj_list_last

        wordCount=[]
        for i in range(len(arr_list)):
            for j in range(len(arr_list[i])):
                wordCount.append(arr_list[i][j])
        # 영상 합치기
        # 데이터프레임 
        wordData=pd.read_csv('Data_Deep/word_data.csv')

        # 데이터프레임에 있는 json 과 단어를 뽑아서 2차원 리스트로 만들기
        wordList = []
        for i in range(len(wordData)):
            jsonList=[]
            for j in range(1):
                jsonList.append(wordData.iloc[i,1])
                jsonList.append(wordData.iloc[i,2])
                jsontuple = tuple(jsonList)
            wordList.append(jsontuple)

        # 2차원 리스트 ( wordList )안에 샘플데이터 ( testList ) 가 있는지 확인
        jsonList2 = []
        for i in range(len(wordList)):
            if wordList[i][0] in wordCount:
                jsonList2.append(wordList[i][1]) # 맞는 번호의 json파일 

        jsonFileName=[]
        json_data=[]
        for i in range(len(jsonList2)):
            # 3. json파일 오픈
            jsonMovieData=[]
            for j in range(1):
                with open('Data_Deep/3000/'+jsonList2[i],'r',encoding='utf-8') as f:
                    json_data.append(json.load(f))
                    #print(json.dumps(json_data))
                    jsonMovieData.append(json_data[i]['metaData']['name'])
                    jsonMovieData.append(json_data[i]['data'][0]['start'])
                    jsonMovieData.append(json_data[i]['data'][0]['end'])
            jsonFileName.append(jsonMovieData)


        clips = []
        try:
            for i in range(len(jsonFileName)):
                mov = VideoFileClip('Data_Deep/Wordmp4/real_word_3000/'+jsonFileName[i][0]).subclip(jsonFileName[i][1],jsonFileName[i][2])
                mov = mov.resize(height=1080,width=1920) # 크기 맞추기
                clips.append(mov)
                print('성공')
        except:
            print('skip')
        print('last',clips)
        path = 'sua8.mp4'
        final_clip = concatenate_videoclips(clips, method='compose')
        final_clip.write_videofile('C:/Users/smhrd/git/BRIDGE_spring/Signal/src/main/webapp/WEB-INF/video/'+path)

        
        return redirect("http://localhost:8082/web/detailpage?link="+path)
    
if __name__ == '__main__':
    app.run(host= '61.80.106.115', port=3306)
# -

# # 테스트


        # 네이버 api 연동 및 경로 설정
res = ClovaSpeechClient().req_upload(file='C:/Users/smhrd/Desktop/Machine Learning/test/data/'+title+'.3gpp', completion='sync')

file='C:/Users/smhrd/Desktop/Machine Learning/test/data/'+title+'.3gpp'
print(yt.title)
        

# +

# 전체 텍스트를 json 타입 변수에 저장
# 텍스트 추출
text = json.loads(res.text)

# kss 활용 텍스트 문장 화
word_list = kss.split_sentences(text['text'])

word_list=word_list[0:5]
word_list

# 문장 끝 마침표 제거

for i in range(len(word_list)):
    
    word_list[i]=word_list[i].replace('.','')

    # 명사만 가져오기 위한 삭제
okt = Okt()
headline = []
stopwords = [ '의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','등','으로도']
for sentence in word_list:
    temp = []
    # morphs() : 형태소 단위로 토큰화
    # stem = True : 형태소에서 어간을 추출
    temp = okt.morphs(sentence, stem = True)
    temp = [word for word in temp if not word in stopwords]
    headline.append(temp)

   # konlpy 트위터 이용 형태소 분류
twitter = Twitter()
sentences_tag_last=[]
for word in headline:
    sentences_tag=[]
    for i in word :
        morph = twitter.pos(i)
        sentences_tag.append(morph)
    sentences_tag_last.append(sentences_tag)

#  형태소 분류
adj_list_last=[]
for ko in sentences_tag_last:
    noun_adj_list=[]
    for i in ko:
        for word, tag in i:
            if tag in ['Noun','Verb','Number','Adjective','Adverb','Alpha']:
                noun_adj_list.append(word)
    adj_list_last.append(noun_adj_list)

print(adj_list_last)
# 형태소 분류
for i in range(len(adj_list_last)):
    for j in range(len(adj_list_last[i])):
        if lemmatize(adj_list_last[i][j]) != None :
                adj_list_last[i][j] = lemmatize(adj_list_last[i][j])

arr_list = adj_list_last

wordCount=[]
for i in range(len(arr_list)):
    for j in range(len(arr_list[i])):
        wordCount.append(arr_list[i][j])
# 영상 합치기
# 데이터프레임 
wordData=pd.read_csv('Data_Deep/word_data.csv')

# 데이터프레임에 있는 json 과 단어를 뽑아서 2차원 리스트로 만들기
wordList = []
for i in range(len(wordData)):
    jsonList=[]
    for j in range(1):
        jsonList.append(wordData.iloc[i,1])
        jsonList.append(wordData.iloc[i,2])
        jsontuple = tuple(jsonList)
    wordList.append(jsontuple)

# 2차원 리스트 ( wordList )안에 샘플데이터 ( testList ) 가 있는지 확인
jsonList2 = []
for i in range(len(wordList)):
    if wordList[i][0] in wordCount:
        jsonList2.append(wordList[i][1]) # 맞는 번호의 json파일 

jsonFileName=[]
json_data=[]
for i in range(len(jsonList2)):
    # 3. json파일 오픈
    jsonMovieData=[]
    for j in range(1):
        with open('Data_Deep/3000/'+jsonList2[i],'r',encoding='utf-8') as f:
            json_data.append(json.load(f))
            #print(json.dumps(json_data))
            jsonMovieData.append(json_data[i]['metaData']['name'])
            jsonMovieData.append(json_data[i]['data'][0]['start'])
            jsonMovieData.append(json_data[i]['data'][0]['end'])
    jsonFileName.append(jsonMovieData)


clips = []
try:
    for i in range(len(jsonFileName)):
        mov = VideoFileClip('Data_Deep/Wordmp4/real_word_3000/'+jsonFileName[i][0]).subclip(jsonFileName[i][1],jsonFileName[i][2])
        mov = mov.resize(height=1080,width=1920) # 크기 맞추기
        clips.append(mov)
        print('성공')
except:
    print('skip')
print('last',clips)

final_clip = concatenate_videoclips(clips, method='compose')
final_clip.write_videofile('C:/Users/smhrd/Desktop/JavaScript/TestFlask/src/main/webapp/video/sua7.mp4')

path = 'sua7.mp4'

# +
wordCount=[]
for i in range(len(arr_list)):
    for j in range(len(arr_list[i])):
        wordCount.append(arr_list[i][j])
        
print(wordCount)
# -

arr_list

# +

# 문장 끝 마침표 제거
last_list = [] # 문장 하나당 리스트 구축
for i in range(len(word_list)):
    last = []
    last.append(word_list[i].replace('.',''))
    last_list.append(last)

# 명사만 가져오기 위한 삭제
okt = Okt()
headline = []
stopwords = [ '의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','등','으로도']
for sentence in word_list:
    for i in sentence:
        temp = []
        # morphs() : 형태소 단위로 토큰화
        # stem = True : 형태소에서 어간을 추출
        temp = okt.morphs(i, stem = True)
        temp = [word for word in temp if not word in stopwords]
        headline.append(temp)

# konlpy 트위터 이용 형태소 분류
twitter = Twitter()
sentences_tag_last=[]
for word in headline:
    sentences_tag=[]
    for i in word :
        morph = twitter.pos(i)
        sentences_tag.append(morph)
    sentences_tag_last.append(sentences_tag)

# -

yt.streams.all()

# ### 아래코드는 분할 테스트 및 가독용 코드

for i in range(len(adj_list_last)):
    print(adj_list_last[i])

# +
# 형태소 분류
for i in range(len(adj_list_last)):
    for j in range(len(adj_list_last[i])):
        if lemmatize(adj_list_last[i][j]) != None :
                adj_list_last[i][j] = lemmatize(adj_list_last[i][j])

arr_list = adj_list_last
arr_list
# -

text = json.loads(res.text)
text['text']

# #  코드 분할

# +
if __name__ == '__main__':
    # https://www.youtube.com/watch?v=kFnHWpGs-18  :: 스마트 인재 개발원
    # https://www.youtube.com/watch?v=lZi3k_GzfCk  :: 서강대 2:25 ~ 4: 10
    youtube=input('다운로드 받을 유튜브 영상 링크 : ')

    yt = pytube.YouTube(youtube)

    # 유튜브 영상 다운로드 후 저장
    stream = yt.streams.all()[0]
    stream.download(output_path='C:/Users/smhrd/Desktop/Machine Learning/test/data') 
    
    # 영상 제목
    title = yt.title
    # res = ClovaSpeechClient().req_url(url='http://example.com/media.mp3', completion='sync')
    # res = ClovaSpeechClient().req_object_storage(data_key='data/media.mp3', completion='sync')
    res = ClovaSpeechClient().req_upload(file='C:/Users/smhrd/Desktop/Machine Learning/test/data/'+title+'.3gpp', completion='sync')
    #print(res.text)
    text = json.loads(res.text)
    # kss 활용 텍스트 문장 화
    word_list = kss.split_sentences(text['text'])
    
    # 명사만 가져오기 위한 삭제
    okt = Okt()
    headline = []
    stopwords = [ '의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','등','으로도']
    for sentence in word_list:
        temp = []
        # morphs() : 형태소 단위로 토큰화
        # stem = True : 형태소에서 어간을 추출
        temp = okt.morphs(sentence, stem = True)
        temp = [word for word in temp if not word in stopwords]
        headline.append(temp)

    # konlpy 트위터 이용 형태소 분류
    twitter = Twitter()
    sentences_tag = []
    for word in headline:
        for i in word :
            morph = twitter.pos(i)
            sentences_tag.append(morph)
    # print(sentences_tag)
    
    #  형태소 분류
    noun_adj_list=[]
    for i1 in sentences_tag:
        for word, tag in i1:
            if tag in ['Noun','Verb','Number','Adjective','Adverb','Alpha']:
                noun_adj_list.append(word)
    # print(noun_adj_list)

    # 형태소 분류
    for i in range(len(noun_adj_list)):
        #print(lemmatize(noun_adj_list[i]))
        if lemmatize(noun_adj_list[i]) != None :
            noun_adj_list[i] = lemmatize(noun_adj_list[i])
            #print(noun_adj_list)

    arr_list = noun_adj_list
    print(arr_list)

    # 영상합치기 부분으로 넘어가기
        
    # 데이터프레임 
    wordData=pd.read_csv('Data_Deep/word_data.csv')
    
    # 데이터프레임에 있는 json 과 단어를 뽑아서 2차원 리스트로 만들기
    wordList = []
    for i in range(len(wordData)):
        jsonList=[]
        for j in range(1):
            jsonList.append(wordData.iloc[i,1])
            jsonList.append(wordData.iloc[i,2])
            jsontuple = tuple(jsonList)
        wordList.append(jsontuple)
        
    # 2차원 리스트 ( wordList )안에 샘플데이터 ( testList ) 가 있는지 확인
    jsonList2 = []
    for i in range(len(wordList)):
        if wordList[i][0] in testList:
            #print(wordList[i][0])
            jsonList2.append(wordList[i][1]) # 맞는 번호의 json파일 

    #print(jsonList2)
    
jsonFileName=[]
json_data=[]
for i in range(len(jsonlist)):
    # 3. json파일 오픈
    jsonMovieData=[]
    for j in range(1):
        with open('Data_Deep/3000/'+jsonlist[i],'r',encoding='utf-8') as f:
            json_data.append(json.load(f))
            #print(json.dumps(json_data))
            jsonMovieData.append(json_data[i]['metaData']['name']) # 단어 이름
            jsonMovieData.append(json_data[i]['data'][0]['start']) # 시작
            jsonMovieData.append(json_data[i]['data'][0]['end']) # 끝
    jsonFileName.append(jsonMovieData)
        
        
clips = []
try:
    for i in range(len(jsonFileName)):
        mov = VideoFileClip('Data_Deep/Wordmp4/real_word_3000/'+jsonFileName[i][0]).subclip(jsonFileName[i][1],jsonFileName[i][2])
        mov = mov.resize(height=1080,width=1920) # 크기 맞추기
        clips.append(mov)
        print('성공')
except:
    print('skip')
print('last',clips)
    
final_clip = concatenate_videoclips(clips, method='compose')
final_clip.write_videofile('Success/'+title+'.mp4')
# -

print(arr_list)



text3= json.loads(res.text)
text3

listtext3 = kss.split_sentences(text3['text'])

# +

last3_list=[]
for i in range(len(listtext3)):
    last3 = []
    last3.append(listtext3[i].replace('.',''))
    last3_list.append(last3)
    
# -

last3_list

testClip = mp.VideoFileClip('Success/python 02.mp4')
testClip_resized = testClip.resize(height=360)
testClip_resized.write_videofile('Success/python 02 resized.mp4')



# +
testClips=[]
mov = VideoFileClip('Success/과정.mp4')
mov = mov.resize(height=1080,width=1920)
testClips.append(mov)
mov2 = VideoFileClip('Success/메신저 (messenger).mp4')
mov2 = mov2.resize(height=1080,width=1920)
testClips.append(mov2)

final_clip = concatenate_videoclips(testClips, method='compose')
final_clip.write_videofile('Success/finalTestClip2.mp4')
# -

print(mov.w)
print(mov.h)
print(mov2.w)
print(mov2.h)


