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

# # 라이브러리 및 모델 불러오기

from flask import Flask ,render_template
from flask import request, redirect
from konlpy.tag import Komoran # 형태소 분석 라이브러리
from konlpy.tag import Twitter # 형태소 분석 라이브러리
from konlpy.tag import Kkma,Okt
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



# +
# https://www.youtube.com/watch?v=kFnHWpGs-18
youtube=input('다운로드 받을 유튜브 영상 링크 : ')

yt = pytube.YouTube(youtube)

title = yt.title
# -



stream = yt.streams.all()[0]
stream.download(output_path='C:/Users/smhrd/Desktop/Machine Learning/test/data')

yt.title

clip = mp.VideoFileClip("data/test1.3gpp")
newsound = clip.subclip("00:01:10","00:01:30")
newsound.audio.write_audiofile("data/audio5.wav",16000,2,2000,'pcm_s16le')

# +
# 오디오 파일 로드
filename = "data/audio5.wav"

r = sr.Recognizer()

text = []
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    
    text = r.recognize_google(audio_data,language='ko-KR')
    print(text)


# +
# 형태소 구분 함수
def lemmatize(word):
    morphtags = Komoran().pos(word)
    if morphtags[0][1] == 'NNG' or morphtags[0][1] == 'NNP':
        return morphtags[0][0]
    


# +
word = '앞서 전화 드린 것처럼 지금부터 이른바 청와대 비서실 새로 지목된 최순실 관련 소식을 집중보도 하겠습니다 지난주 JTBC는 최순실 씨 최측근이라고 하는 고영태 씨를 주 대한 내용을 단독으로 보내 드렸습니다 최순실 씨가 유일하게 잘하는 것이 대통령 연설문 수정하는 것이다라는 내용이었는데요이 내용을 보도하자 청와대 이원종 비서실장은 정상적인 사람이면 믿을 수 있겠느냐 공부할 때도 있을 수 없는 얘기 다 이렇게 얘기 한 바 있습니다 JTBC가 몇 시에 말을 보도한 배경에는 사실 또 다른 믿기 어려운 정황이 있기 때문에 왔습니다 JTBC 취재팀은 최순실 씨 컴퓨터 입수해서 분석을 했습니다 태식아 대통령 연설문 에바다 봤다는 사실을 확인할 수 있었습니다 그런데 3시가 연설문 44개를 파일형태로 받은 실점을 너무도 대통령이 연설을 하기 의견이었습니다 먼저 김필중 기자의 단독 벌 벌 벌 이어가겠습니다 제수씨 사무실에 있던 pc에 저장된 파일들입니다 각종 문서도 가득합니다 파일은 모두 200여개의 일입니다 그런데 씨가 보관 중인 파일에 대부분이 청와대 와 관련된 내용이었습니다 기회 되면 특히 제시가 대통령 연설문 수정했다는 최초의 책은 고영태의 준수과 관련해 연설문의 주목했습니다 첼시가 갖고 있던 연설문 또는 공식 빠른한테 파일엔 모드 44개 없습니다 대선후보 시절에 박 대통령의 유선문의 비롯해 대통령 취임 연설문 드리기로 했습니다 그런데 최씨가 이문권 에바다 여러분 시장은 대통령이 실제 발언했던 것보다 길게는 4월이나 없었습니다 상당수 대통령 연설문 사전에 청와대 내부에서도 도움이 되지 않는다는 점을 감안하면 연설문이 사전에 청와대 아무거나 최씨에게 전달되었던 사실은 이른바 비선실세 놀란 거 관련해서 큰 날 것으로 보입니다 JTBC 김필 주입니다'

word_list = kss.split_sentences(text)

from collections import Counter

okja = []

for line in word_list:
    okja.append(line)
print(okja)

print('------------------------------------------------------------')

twitter = Twitter()
sentences_tag = []
for i in okja:
    morph= twitter.pos(i)
    sentences_tag.append(morph)
    
print(sentences_tag)

# -

noun_adj_list=[]
for i1 in sentences_tag:
    for word, tag in i1:
        if tag in ['Noun','Verb','Number','Adjective','Adverb']:
            noun_adj_list.append(word)
print(noun_adj_list)

for i in range(len(noun_adj_list)):
    morphtags = Komoran().pos(noun_adj_list[i])
    print(noun_adj_list[i])
    print('---------')
    print(morphtags)
    print('=========')
    print(lemmatize(noun_adj_list[i])) # NNG, NNP

# +
for i in range(len(noun_adj_list)):
    print(lemmatize(noun_adj_list[i]))
    if lemmatize(noun_adj_list[i]) != None :
        noun_adj_list[i] = lemmatize(noun_adj_list[i])
        print(noun_adj_list)
        
arr_list = noun_adj_list
print(arr_list)

 # 영상합치기 부분으로 넘어가기
# -


