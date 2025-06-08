if __name__ == "__main__":
    #region 임포트 & 준비
    from utils import *
    from main_func import process_data, log, error, save, multi_decorator
    #endregion
    
    log("시작")

    #region 데이터 준비
    try:
        log("데이터 준비중...")

        texts = [
            "오늘 너무 즐거운 하루였어", "기분이 상쾌하다", "행복한 기분이 드는 날이야", "짜증나는 일이 또 생겼어", "정말 최악의 하루야", "화가 머리끝까지 났다", "기분이 좋지는 않네", "웃음이 나는 좋은 날", "오늘은 운이 좋은 날이야",    "속상하고 우울해", "너무 답답하고 슬퍼", "화나는 일이 있어서 미치겠어", "편안한 하루를 보냈어", "오늘은 참 평화로웠어", "기분이 안정된다", "짜증이 났지만 참고 넘겼어", "분위기가 좋아서 웃음이 났어", "기분이 좋아서 노래를 불렀어",    "모두 나를 무시하는 것 같아", "억울하고 속상해", "왜 이렇게 힘든 거야", "행복한 일이 생겼어", "오늘 고마운 일이 있었어", "친절한 사람을 만나서 기뻤어", "무기력하고 아무것도 하기 싫다", "몸도 마음도 너무 피곤해", "우울해서 아무 말도 하기 싫어",    "사랑하는 사람과 좋은 시간을 보냈어", "이런 날씨 너무 좋아", "맛있는 거 먹고 기분이 좋아졌어",    "일이 다 망한 것 같아", "다 끝장난 느낌이야", "절망스럽고 괴로워","오늘 하루가 완벽했어", "운동하고 나니 기분 최고", "재밌는 영화 보고 왔어",    "진짜 어이없는 상황이었어", "실망이 너무 커", "분노가 치밀어 올라",    "마음이 차분해지고 좋아", "힐링되는 시간이었어", "산책하면서 기분 전환했어",    "혼자 있다는 게 너무 외로워", "눈물이 날 것 같아", "불안하고 초조해",    "오랜만에 친구 만나서 즐거웠어", "좋은 음악 들으니까 행복해", "기분이 좋아서 춤췄어",    "다신 이런 일 겪고 싶지 않아", "가슴이 답답하고 괴로워", "아무도 날 이해하지 못해"
        ]
        labels = [
            1, 1, 1,
            0, 0, 0,
            0, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1,
            0, 0, 0
        ]  # 1: 긍정, 0: 부정
        results= []
        
        log("데이터 준비 완료")
    except Exception as e:    
        error("데이터 준비")
    #endregion
    using_analyzer = "Hannanum"
    #region 전처리(정규화 등) (토큰화)
    try:
        log(f"전처리 준비중... (분석기: {using_analyzer})")

        import preprocessing
        
        pre_steps = [
            multi_decorator(preprocessing.clean_text), #정규화
            multi_decorator(preprocessing.analyzers[using_analyzer].pos), #형태소 분석
            multi_decorator(preprocessing.posToStr),
            multi_decorator(preprocessing.non_meaning_remove) #불용어
        ]

        log("전처리 중...")
        pre_texts = process_data(texts, pre_steps)
        log("전처리 완료")
    except Exception as e:    
        error("전처리")
    #endregion
    #region 전처리 데이터 저장 (토큰 전처리)
    try:
        log("전처리 데이터 저장중...")
        save(labels, pre_texts, f"{using_analyzer}_preprocessing.txt")
        log("전처리 데이터 저장 완료")
    except Exception as e:    
        error("전처리 데이터 저장")
    #endregion
    using_extraction = "BoW"
    #region 특징 추출 (벡터화)
    try:
        log(f"특징 추출 준비중... (사용 방법: {using_extraction})")

        import feature_extraction

        extraction_pre_steps = {
            "BoW": [
                multi_decorator(feature_extraction.vocabularyCreate), #사전 제작
            ]
        }
        extraction_steps = {
            "BoW": [
                multi_decorator(feature_extraction.document_to_vector)
            ]
        }

        log("특징 추출 중...")
        extracton_pre_texts = process_data(pre_texts, extraction_pre_steps[using_extraction])
        extracton_texts = process_data(extracton_pre_texts, extraction_steps[using_extraction])
        log("특징 추출 완료")
    except Exception as e:    
        error("특징 추출")
    #endregion
    #region 특징 추출 데이터 저장 
    try:
        log("특징 추출 데이터 저장중...")
        save(labels, extracton_texts, f"{using_analyzer}_{using_extraction}_featureExtraction.txt")
        log("특징 추출 데이터 저장 완료")
    except Exception as e:    
        error("특징 추출 데이터 저장")
    #endregion
    random_state = 42
    test_size = 0.2
    using_model = "SVC-linear"
    #region 학습
    try:
        #학습 준비
        log(f"학습 준비중... (랜덤 시드: {random_state})")
        X = np.array(extracton_texts).reshape(len(extracton_texts), -1)
        labels = np.array(labels).reshape(-1)
        
        # 학습/테스트 분할
        log(f"데이터 분할 중... (테스트 데이터 비율: {test_size})")
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=random_state, stratify=labels) 

        # SVM 모델 학습
        log(f"학습 중... (모델 이름: {using_model})")
        models = {
            #SVC
            "SVC-linear" : SVC(kernel='linear'),
            "SVC-rbf" : SVC(kernel='rbf'),
            "SVC-poly" : SVC(kernel='poly'),
            "SVC-sigmoid" : SVC(kernel='sigmoid'),
            #
        }
        svm_model = deepcopy(models[using_model]) 
        svm_model.fit(X_train, y_train)
        log("학습 완료")
    except Exception as e:    
        error("학습")
    #endregion
    #region 평가
    try:
        # 평가
        log("평가 중...")
        y_pred = svm_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("=== 평가 결과 ===")
        print(report)
        log("평가 완료")
    except Exception as e:    
        error("평가")
    #endregion
    new_text = ["기분이 정말 좋아", "너무 짜증난다"]
    #region 새 문장 예측
    try:
        # 새 문장 예측을 위한 pipeline
        t = f"새 문장 준비 중... (\"{new_text[0]}\""
        if len(new_text) >= 2:
            t += f" ... \"{new_text[-1]})\""
        else:
            t += ")"
        log(t)

        getVec_steps = pre_steps + extraction_steps[using_extraction] #전처리 + 벡터화

        # 새 문장 예측
        log("새 문장 예측중...")
        new_vec = process_data(new_text, getVec_steps) #벡터로 변환

        new_vec = np.array(new_vec).reshape(2, -1)
        predictions = svm_model.predict(new_vec)

        #예측값 출력
        log("새 문장 예측완료")
        for text, pred in zip(new_text, predictions):
            label = "긍정" if pred == 1 else "부정"
            print(f'"{text}" → {label}')
        
        #사용자 입력 예측하기
        log("사용자 입력 예측 중...")
        while "" != (s := input()):
            new_vec = process_data([s], getVec_steps) #벡터로 변환
            new_vec = np.array(new_vec).reshape(1, -1)
            predictions = svm_model.predict(new_vec)

            label = "긍정" if predictions[0] == 1 else "부정"
            print(f'"{s}" → {label}')
    except Exception as e:    
        error("예측")
    #endregion

    log("종료\n")

# 계획
#(보류)1. 데이터를 불러온다 (파일에서)
#(완료)2. 전처리를 한다. (다른 파일에 저장하기)
#(완료)3. 특징추출을 한다(다른 파일에 저장하기) (이 데이터로 학습을 진행)
#(완료)4. 학습 시킨다.
#(완료)5. 테스트 한다.

#TODO
#데이터 준비
#모델 저장
#특징 추출 방법 추가
#ui 만들기
