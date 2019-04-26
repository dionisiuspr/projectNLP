#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[1]:


import numpy as np
import pandas as pd
import re

#Natural Language Toolkit
import nltk
from nltk import NaiveBayesClassifier
from nltk import DecisionTreeClassifier
from nltk import MaxentClassifier
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import brown
from nltk.tag import UnigramTagger

#Pickle
import pickle


# ## Paths

# In[2]:


URL_TRAIN = "train_01.csv"
URL_TEST = "test_01.csv"

#Emotions/Expression GAZ path
ANGER_WORD = "gaz/anger.txt"
FEAR_WORD = "gaz/fear.txt"
HAPPY_WORD = "gaz/happy.txt"
SAD_WORD = "gaz/sad.txt"

#Text type GAZ path
NEG_WORD = "gaz/negative-words.txt"
POS_WORD = "gaz/positive-words.txt"

#Pickle location
DIR_PICKLE = "pickle/train.pickle"


# ## Labels

# ## Pickle Manager

# In[3]:


class PickleManager():
    @staticmethod
    def save_pickle(object, path):
        pickle.dump(object, open(path, "wb"))
        
    
    @staticmethod
    def open_pickle(path):
        return pickle.load(open(path, "rb"))


# # Data Elements

# In[4]:


class DataElement():
    def __init__ (self, document, index, label = None):
        self.raw_text = document
        self.label = label
        self.clean_text = Preprocessing.do_preprocessing(self.raw_text)
        self.index = index
    
    def get_data_str(self):
        print ('='*30)
        print("Doc num/Index :",self.index)
        print("Raw Doc :", self.raw_text)
        print("Clean Doc :",self.clean_text)
        print("Result :",self.label)


# # Preprocessing

# In[5]:


class Preprocessing():
    @staticmethod
    def do_preprocessing(document):
        document = document.strip()
        document = document.lower()
        document = Preprocessing.text_preprocessing(document)
        return document
    
    @staticmethod
    def text_preprocessing(sent):
        sent = re.sub("@\\w+|#\\w+|\\bRT\\b","",sent)
        sent = re.sub("https?://\\S+\\s?","<LINK>",sent)
        sent = re.sub("[ ]+", " ",sent)
        return sent


# # Feature Set Extraction

# In[6]:


class FeaturesetExtractor():

    def __init__(self):
        self.neg_words = [line.rstrip('\n') for line in open(NEG_WORD)]
        self.pos_words = [line.rstrip('\n') for line in open(POS_WORD)]
        self.anger_words = [line.rstrip('\n') for line in open(ANGER_WORD)]
        self.fear_words = [line.rstrip('\n') for line in open(FEAR_WORD)]
        self.happy_words = [line.rstrip('\n') for line in open(NEG_WORD)]
        self.sad_words = [line.rstrip('\n') for line in open(SAD_WORD)]
        self.tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
        
    def get_featureset(self, data_element):
        mapFeatureset = {}
        size = len(data_element.clean_text)
        word = data_element.clean_text
        list_word = word.split(" ")
        raw = data_element.raw_text
        list_word_raw = raw.split(" ")
        
        tot_pos_words = len(set(list_word) & set(self.pos_words))
        tot_neg_words = len(set(list_word) & set(self.neg_words))
        
        list_anger = tuple(set(list_word) & set(self.anger_words))
        list_fear = tuple(set(list_word) & set(self.fear_words))
        list_happy = tuple(set(list_word) & set(self.happy_words))
        list_sad = tuple(set(list_word) & set(self.sad_words))

        exclamation_count = raw.count("!")
        question_count = raw.count("?")
        uppercase_count = sum(1 for c in raw if c.isupper())

        mapFeatureset["bias"] = 1
        mapFeatureset["word"] = tuple(list_word)
        mapFeatureset["neg_words"] = tot_neg_words
        mapFeatureset["pos_words"] = tot_pos_words
        mapFeatureset["exclamation_count"] = exclamation_count
        mapFeatureset["question_count"] = question_count
        mapFeatureset["list_happy"] = list_happy
        mapFeatureset["list_sad"] = list_sad
        mapFeatureset["list_fear"] = list_fear
        mapFeatureset["list_anger"] = list_anger
        
        pos_tag_temp = self.tagger.tag((word).split(" "))
        list_pos_tag = []
        for element in pos_tag_temp:
            list_pos_tag.append(element[1])
        mapFeatureset["pos_tag"] = tuple(list_pos_tag)
        
        return mapFeatureset   


# ## Text Classification

# In[7]:


class TextCategorization():
    def __init__(self, list_data_element, feature_extractor, classifier = "maxent"):
        self.classifier = classifier
        self.model = self.get_model()
        self.featuresets = []
        self.list_data_element = list_data_element
        self.feature_extractor = feature_extractor
    
    def get_model(self):
        model = None
        self.classifier = ""
        if self.classifier == "decision_tree":
            model = DecisionTreeClassifier
        elif self.classifier == "maxent":
            model = MaxentClassifier
        else:
            model = NaiveBayesClassifier
        return model
        
    def build_model(self):
        print("Build Model function")
        self.get_featuresets()
        self.classifier = self.model.train(self.featuresets)
    
    def get_classify(self, data_element):
        featureset = self.feature_extractor.get_featureset(data_element)
        result = self.classifier.classify(featureset)
        return result   

    def get_featuresets(self):
        print("Get Featureset...")
        for data_element in self.list_data_element :
            featureset = self.feature_extractor.get_featureset(data_element)
            self.featuresets.append((featureset, data_element.label))  


# ## Main Wrapper 

# In[8]:


def get_list_data_element(url):
    #load data
    df_data_element = pd.read_csv(url,encoding = "ISO-8859-1").dropna()
    list_data_element = []
    index = 0
    for index, row in df_data_element.iterrows():
        document = row['document']
        label = row['label']
        data_element = DataElement(document, index, label)
        list_data_element.append(data_element)
    return list_data_element


# In[9]:


def do_learning():
    list_data_element = get_list_data_element(URL_TRAIN)
    feature_extractor = FeaturesetExtractor()
    textCat = TextCategorization(list_data_element, feature_extractor, classifier = "maxent")
    textCat.build_model()
    
    print("Saving Pickle!")
    PickleManager.save_pickle(textCat, DIR_PICKLE)
    print("Saved!")


# In[10]:


def do_test_from_input(hasil):
    textCat = PickleManager.open_pickle(DIR_PICKLE)
    data_element = DataElement(hasil, 0)
    result = textCat.get_classify(data_element)
    data_element.get_data_str()
    return result


# In[ ]:


import json
import os
import requests
import datetime
from datetime import timedelta

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import auth

from googletrans import Translator

from flask import Flask
from flask import request
from flask import make_response

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

# firebase
cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://treat-me-hackathon.firebaseio.com/'
})


# Flask app should start in global layout
app = Flask(__name__)

line_bot_api = LineBotApi('srPYwTexW+Tr1MPKXVdLL3qcuAkXjz5kIN3FR8BrXBzFNWEmJiaZWFFz54lV4Y9fGTGQ3x8PlSyE7qnhFGFaDHW+1yRAv9uwRJ4kDcUJM68x3ygfvpujcXY5sni0sm/UJbk1lGVZWGKFWpkAlVrRuwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('55a16528cf419b682e5f67f5ed225da1')

@app.route('/call', methods=['GET'])


def call():
    translator = Translator()
    database = db.reference()
    user = database.child("user")

    #membaca apakah ada data pada firebase

    snapshot = user.order_by_key().get()
    #key = userId Line
    d=""
    for key, val in snapshot.items():
        try:
            if str(val["stat"])=="2":
                lMessage= user.child(str(val["connect"])).child("lastMessage").get()
                lName= user.child(str(val["connect"])).child("name").get()
                #push message jika User memiliki lastMessage
                if lMessage!=None:
                    hasil = str(translator.translate(str(lMessage), src="id",dest="en").text)
                    hasil_klasifikasi = str(do_test_from_input(hasil))
                    line_bot_api.push_message(key, TextSendMessage(text="Emosi pasien "+str(lName)+" selama 5 menit terakhir terdeteksi sebagai : "+str(hasil_klasifikasi)+"\n\nSumber : "+str(hasil)))
#                     return do_test_from_input(hasil)
                    #untuk reset
                    reset = user.child(str(val['connect']))
                    reset.update({
                        "lastMessage" : None
                    })
        except Exception as res:
            d = d+"\n"+str(res)
    return d



        
    
    
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 4040))

    print ("Starting app on port %d" %(port))

    app.run(debug=False, port=port, host='0.0.0.0')


# ## Main

# In[11]:


# do_learning()


# In[ ]:




