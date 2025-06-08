if __name__ == "__main__":
    ###임포트 & 준비
    from utils import *
    from main_func import process_data, log, error, save, multi_decorator
    
    log("시작")
    ###데이터 준비
    try:
        log("데이터 준비중...")

        texts   = ["안녕하세요. KoNLPy를 사용한 형태소 분석 예제입니다.", "안녕하세요! 이 문구는 테스트용 입니다!", "잘있어 친구야", "hello, everyone? nice to meet you!", "죽다 죽기 죽음 죽고", "나는 사과를 먹는다, 순댓국"]
        labels  = [1,0,1,-1,0, 0] # 1: 긍정, 0: 중립, -1: 부정
        results= []
        
        log("데이터 준비 완료")
    except Exception as e:    
        error("데이터 준비")
    ###전처리(정규화, 토큰화 등) (str -> list)
    try:
        using_analyzer = "Hannanum"
        log(f"전처리 준비중... (분석기: {using_analyzer})")

        import preprocessing
        
        pre_steps = [
            multi_decorator(preprocessing.clean_text), #정규화
            multi_decorator(preprocessing.analyzers[using_analyzer].pos), #형태소 분석
            multi_decorator(preprocessing.posToStr),
            multi_decorator(preprocessing.non_meaning_remove) #불용어
        ]

        log("전처리 중...")
        texts = process_data(texts, pre_steps)
        log("전처리 완료")
    except Exception as e:    
        error("전처리")
    ###전처리 데이터 저장
    try:
        log("전처리 데이터 저장중...")
        save(labels, texts, f"{using_analyzer}_preprocessing.txt")
        log("전처리 데이터 저장 완료")
    except Exception as e:    
        error("전처리 데이터 저장")
    ###특징 추출
    try:
        using_extraction = "BoW"
        log(f"특징 추출 준비중... (사용 방법: {using_extraction})")

        import feature_extraction

        extraction_steps = {
            "BoW": [
                multi_decorator(feature_extraction.vocabularyCreate), #사전 제작
                multi_decorator(feature_extraction.document_to_vector)
            ]
        }

        log("특징 추출 중...")
        texts = process_data(texts, extraction_steps[using_extraction])
        log("특징 추출 완료")
    except Exception as e:    
        error("특징 추출")
    ###특징 추출 데이터 저장
    try:
        log("특징 추출 데이터 저장중...")
        save(labels, texts, f"{using_analyzer}_{using_extraction}_featureExtraction.txt")
        log("특징 추출 데이터 저장 완료")
    except Exception as e:    
        error("특징 추출 데이터 저장")
    ###학습
    log("종료\n")

# 계획
#(보류)1. 데이터를 불러온다 (파일에서)
#(완료)2. 전처리를 한다. (다른 파일에 저장하기)
#(방법 1개 완료)3. 특징추출을 한다(다른 파일에 저장하기) (이 데이터로 학습을 진행)
#4. 학습 시킨다. (피클? 로 저장?)
#5. 테스트 한다. (로그 남기기)
