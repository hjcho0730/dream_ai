from utils import *

#region 정규화-특수문자(마침표 등) 제거
def clean_text(text): 
    _text = deepcopy(text)
    if type(_text) == type("string"):    
        _text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", _text)  # 한글, 영문, 숫자, 공백 제외 모두 제거
        _text = re.sub(r"\s+", " ", _text).strip()  # 연속 공백 1개로 치환
    elif type(_text) == type(list()):
        for i in range(len(_text)):
            _text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", _text)  # 한글, 영문, 숫자, 공백 제외 모두 제거
            _text = re.sub(r"\s+", " ", _text).strip()  # 연속 공백 1개로 치환
    return _text
#endregion

#region 형태소 추출

analyzers: Final[Any] = { ##여기서 원하는 거 쓰면 됨
    "Komoran": Komoran(max_heap_size=BigNum),
    "Okt": Okt(max_heap_size=BigNum),
    "Kkma": Kkma(max_heap_size=BigNum),
    "Hannanum": Hannanum(max_heap_size=BigNum),
}

def posToStr(pos: list[tuple[str]]) -> list[tuple[str]]:
    _pos = pos[:]
    for i in range(len(_pos)):
        _pos[i] = (_pos[i][1] + "_____")[:5]+_pos[i][0]
    return _pos
#endregion

# (임시) 불용어
#한나눔을 위한 불용어
non_meaning_kor: Final[list[str]] = []#['J_은', 'J_는', 'J_이', 'J_가', 'J_을', 'J_를', 'J_에', 'J_에서', 'J_의', 'J_으로', 'J_도', "J_으로", "J_로", "J_에", "J_에서"]
non_meaning_eng: Final[list[str]] = []
def non_meaning_remove(text):
    _text = deepcopy(text)
    if type(_text) == type("string"):
        for i in chain(non_meaning_kor, non_meaning_eng):
            _text = _text.replace(i, "")
    elif type(_text) == type(list()):
        for i in chain(non_meaning_kor, non_meaning_eng):
            if i in _text:
                _text.remove(i)
    return _text

def getNoramlPipeline(using_analyzer):
    from main_func import multi_decorator
    pre_steps = [
        multi_decorator(clean_text), #정규화
        multi_decorator(analyzers[using_analyzer].pos), #형태소 분석
        multi_decorator(posToStr),
        # multi_decorator(preprocessing.non_meaning_remove) #불용어
    ]

    return pre_steps
