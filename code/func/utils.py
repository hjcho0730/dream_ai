import time, os
from itertools import chain

from copy import deepcopy

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix, lil_matrix
#의존성 관리를 위해 외부 라이브러리 및 내장 라이브러리 만 임포트
#2곳 이상에 쓰이는 것들만 임포트 

real_path = os.path.dirname(os.path.realpath(__file__))
real_path = os.path.dirname(os.path.dirname(real_path))
log_path = os.path.join(real_path, "data", "log.txt ")

vocabulary = set()

class DataDict:
    vocabularyDict = dict()
    def getDict(self):
        return self.vocabularyDict
    def setDict(self, d):
        self.vocabularyDict = d
    def deleteVal(self, v):
        self.vocabularyDict.pop(v)
DDD = DataDict()

class DataTransformer:
    tfidf = TfidfTransformer()
    def set(self, trans):
        self.tfidf = trans
tfidf = DataTransformer()

def toArray(t):
    X = csr_matrix(t)
    return X
