from utils import *

#region 정규화-특수문자(마침표 등) 제거
def clean_text(text): 
    if type(text) == type("string"):    
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)  # 한글, 영문, 숫자, 공백 제외 모두 제거
        text = re.sub(r"\s+", " ", text).strip()  # 연속 공백 1개로 치환
    elif type(text) == type(list()):
        for i in range(len(text)):
            text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)  # 한글, 영문, 숫자, 공백 제외 모두 제거
            text = re.sub(r"\s+", " ", text).strip()  # 연속 공백 1개로 치환
    return text
#endregion

#region 형태소 추출

analyzers: Final[Any] = { ##여기서 원하는 거 쓰면 됨
    "Komoran": Komoran(),
    "Okt": Okt(),
    "Kkma": Kkma(),
    "Hannanum": Hannanum()
}


def posToStr(pos):
    for i in range(len(pos)):
        pos[i] = (pos[i][1] + "__")[:2]+pos[i][0]
    return pos
#endregion

# (임시) 불용어
#한나눔을 위한 불용어
non_meaning_kor: Final[list[str]] = ['J_은', 'J_는', 'J_이', 'J_가', 'J_을', 'J_를', 'J_에', 'J_에서', 'J_의', 'J_으로', 'J_도', "J_으로", "J_로", "J_에", "J_에서"]
non_meaning_eng: Final[list[str]] = []
def non_meaning_remove(text):
    if type(text) == type("string"):
        for i in chain(non_meaning_kor, non_meaning_eng):
            text = text.replace(i, "")
    elif type(text) == type(list()):
        for i in chain(non_meaning_kor, non_meaning_eng):
            if i in text:
                text.remove(i)
    return text
