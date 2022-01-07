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

# +
# 라이브러리 및 모델 불러오기

from flask import Flask ,render_template
from flask import request, redirect
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
# BOW = BAG of WORD : 단어가방, 단어모음, 단어사전
from sklearn.feature_extraction.text import CountVectorizer
# 위 도구는 빈도수 기반 벡터화 도구

import json

# # 알고리즘 시작

# # 형태소 구분하는 함수
# - 사용해야할 품사가 생각보다 많음 ( komoran 기준 ) 
#   - 명사 NN -> 일반명사 NNG // 고유명사 NNP // 의존명사 NNB

# 형태소 구분 함수
def lemmatize(word):
    morphtags = Komoran().pos(word)
    if morphtags[0][1] == 'NNG' or morphtags[0][1] == 'NNP':
        return morphtags[0][0]


# + endofcell="--"

# # +
# https://www.youtube.com/watch?v=kFnHWpGs-18  :: 스마트 인재 개발원
# https://www.youtube.com/watch?v=lZi3k_GzfCk  :: 서강대 2:25 ~ 4: 10
youtube=input('다운로드 받을 유튜브 영상 링크 : ')

yt = pytube.YouTube(youtube)

# 영상 제목
title = yt.title
# -

# --

# + endofcell="--"


# 유튜브 영상 다운로드 후 저장

stream = yt.streams.all()[0]
stream.download(output_path='C:/Users/smhrd/Desktop/Machine Learning/test/data') 

# 영상을 오디오 파일로 변환 
clip = mp.VideoFileClip("data/"+title+".3gpp")
newsound = clip.subclip("00:02:25","00:04:10") # 20 sec
newsound.audio.write_audiofile("data/"+title+".wav",16000,2,2000,'pcm_s16le')

# 오디오 파일 로드
filename = "data/"+title+".wav"

# 오디오 파일 텍스트 추출
text = []
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data,language='ko-KR')
    # print(text)

# kss 활용 텍스트 문장 화
word_list = kss.split_sentences(text)

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

# -

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
# --

# +
# 영상 합치기
    
clips = []
for i in range(len(arr_list)):
    # print("*****", clips)
    try:
        # arr_list = ['보고', '강의', '목표', '대해', '이야기', '하다', '다음', '수업', '부터', '본격', '적', '파이썬', '언어', '대한', '학습', '시작', '하다', '수', '있다', '파이썬', '언어', '설치', '하다', '과정', '보다', '교육', '점점', '중요하다', '이유', '소프트웨어', '미래', '사회', '요구', '하다', '용량', '속초', '발전']
        mov = VideoFileClip("Data_Deep/30개영상/"+arr_list[i]+".mp4")
        mov = mov.subclip('00:00:01','00:00:04')
        clips.append(mov)
        print("try")
    except:
        print('skip')
        
    print("last", clips)
final_clip = concatenate_videoclips(clips, method='compose') # concat함수를 이용해 비디오를 합치기
final_clip.write_videofile("Success/sua5.mp4")
# -

# # json
# 1. pd.read_csv 로 단어 라벨링 및 json 파일 이름 DataFrame 으로 가져오기
# 2. DataFrame 인덱싱 이용해서 알고리즘 생성
# 3. json 파일 오픈
# 4. 단어사전 ( arr_list ) 와 json 파일 일치하는지 확인
# 5. 영상 합치기
# 6. Falsk로 java에 띄우기

# 데이터프레임 
wordData=pd.read_csv('Data_Deep/word_data.csv')
wordData

# +
# 데이터프레임에 있는 json 과 단어를 뽑아서 2차원 리스트로 만들기
wordList = []

for i in range(len(wordData)):
    jsonList=[]
    for j in range(1):
        jsonList.append(wordData.iloc[i,1])
        jsonList.append(wordData.iloc[i,2])
    wordList.append(jsonList)

# +
# 2차원 리스트 ( wordList )안에 샘플데이터 ( testList ) 가 있는지 확인

jsonlist = []
for i in range(len(wordList)):
    if wordList[i][0] in testList:
        print(wordList[i][0])
        jsonlist.append(wordList[i][1]) # 맞는 번호의 json파일 담기

print(jsonlist)
# -

#
# # 3. json파일 오픈
# with open('Data_Deep/1000/NIA_SL_FS0001_CROWD01_F_morpheme.json','r',encoding='utf-8') as f:  
#     json_data = json.load(f)  
# print(json.dumps(json_data))  

with open('Data_Deep/3000/'+jsonlist[i],'r',encoding='utf-8') as f:
            json_data = json.load(f)

# +
jsonFileName=[]
for i in range(len(jsonlist)):
    # 3. json파일 오픈
    with open('Data_Deep/3000/'+jsonlist[i],'r',encoding='utf-8') as f:
        json_data = json.load(f)
        #print(json.dumps(json_data))
    jsonFileName.append(json_data['metaData']['name'])
    jsonName = json_data['data'][0]['attributes'][0]['name']
    print(jsonFileName,'------' ,jsonName)
    
clips = []
try:
    for i in range(len(jsonFileName)):
        mov = VideoFileClip('Data_Deep/Wordmp4/real_word_3000/'+jsonFileName[i])
        mov = mov.subclip('00:00:01','00:00:04')
        clips.append(mov)
        print('성공')
except:
    print('skip')
print('last',clips)
    
final_clip = concatenate_videoclips(clips, method='compose')
final_clip.write_videofile('Success/'+title+'.mp4')
# +
# 3-1 json 인덱싱
# dict // key : value ?? list // array

jsonFileName = json_data['metaData']['name']
jsonName = json_data['data'][0]['attributes'][0]['name']
print(jsonFileName,'------' ,jsonName)


# +
# dict // key : value ?? list // array

clips = []
for i in range(len(arr_list)):
    
    if arr_list[i] == '하다' :
        print(arr_list[i],i)
        mov = VideoFileClip('Data_Deep/1000/'+json_data['metaData']['name'])
        mov = mov.subclip('00:00:01','00:00:04')
        clips.append[mov]
        