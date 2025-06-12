import time, sys, os

import re
from itertools import chain
from typing import Union, Final, Any
import numpy as np
from pprint import pprint
from copy import deepcopy

import random as rd

import json

from konlpy.tag import Okt, Kkma, Komoran, Hannanum

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

import pickle
import joblib

from scipy.sparse import csr_matrix, lil_matrix

#의존성 관리를 위해 외부 라이브러리 및 내장 라이브러리 만 임포트

real_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(real_path, "data", "log.txt ")

vocabulary = set()
class DataDict:
    vocabularyDict = dict()
    def getDict(self):
        return self.vocabularyDict

    def setDict(self, d):
        self.vocabularyDict = d
DDD = DataDict()

BigNum = 2000000
class DataTransformer:
    tfidf = TfidfTransformer()
    def set(self, trans):
        self.tfidf = trans
tfidf = DataTransformer()
# vector = np.array([0])

def toArray(t):
    # X = []
    # if type(t) == type(list()):
    X = csr_matrix(t)
    # else:
    #     X = t.toarray()
    return X
