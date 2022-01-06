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

title = yt.title
# -

# --

# + endofcell="--"


# 영상 다운로드 경로

stream = yt.streams.all()[0]
stream.download(output_path='C:/Users/smhrd/Desktop/Machine Learning/test/data') 

# 영상 오디오 파일로 변환 
clip = mp.VideoFileClip("data/python 01.3gpp")
newsound = clip.subclip("00:01:10","00:01:30") # 20 sec
newsound.audio.write_audiofile("data/audio5.wav",16000,2,2000,'pcm_s16le')

# 오디오 파일 로드
filename = "data/audio5.wav"

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
        if tag in ['Noun','Verb','Number','Adjective','Adverb']:
            noun_adj_list.append(word)
# print(noun_adj_list)

# 형태소 분류
for i in range(len(noun_adj_list)):
    #print(lemmatize(noun_adj_list[i]))
    if lemmatize(noun_adj_list[i]) != None :
        noun_adj_list[i] = lemmatize(noun_adj_list[i])
        print(noun_adj_list)

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
        clips.append(mov)
        print("try")
    except:
        print('skip')
        
    print("last", clips)
final_clip = concatenate_videoclips(clips, method='compose') # concat함수를 이용해 비디오를 합치기
final_clip.write_videofile("Success/sua5.mp4")
# -



# +
# 테스트

# 영상 다운로드 경로

stream = yt.streams.all()[0]
stream.download(output_path='C:/Users/smhrd/Desktop/Machine Learning/test/data') 

# 영상 오디오 파일로 변환 
clip = mp.VideoFileClip("data/python 01.3gpp")
newsound = clip.subclip("00:01:10","00:01:30") # 20 sec
newsound.audio.write_audiofile("data/audio5.wav",16000,2,2000,'pcm_s16le')

# 오디오 파일 로드
filename = "data/audio5.wav"

# 오디오 파일 텍스트 추출
text = []
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data,language='ko-KR')
    # print(text)

# kss 활용 텍스트 문장 화
word_list = kss.split_sentences(text)
