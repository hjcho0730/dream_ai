import time
import re
import sys, os
#import pprint
from itertools import chain
from typing import Union, Final, Any
from konlpy.tag import Okt, Kkma, Komoran, Hannanum

#의존성 관리를 위해 외부 라이브러리 및 내장 라이브러리 만 임포트

real_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(real_path, "data", "log.txt ")

vocabulary = tuple()
