if __name__ == "__main__":
    #region 임포트 & 준비
    from func.main_func import log
    log("[--모델 학습--]")
    log("임포트 중")
    from  func.utils import *
    from  func.main_func import process_data, error, save, multi_decorator, res
    import numpy as np
    #endregion
    
    #region 설정
    MovieReviewLoadMode= 0 # csv: 0, json: 1, False: 2
    MovieReviewLimit= -1
    mode= ["csv", "json", False]

    save_preData = False           #전처리 데이터 저장 여부
    save_extractionData = False #특징 추출 데이터 저장 여부
    save_model = False#True               #모델 저장 여부

    using_analyzer = "Komoran" #형태소 분석기(전처리)

    using_extraction = "TF-IDF" #특징 추출 방법
    n_gram = True                    # n-gram 여부

    random_state = 42                 #데이터셋 자르기용 난수 설정
    test_size = 0.2                      #테스트셋 비율
    using_model = "SGDClassifier" #모델 이름 설정

    new_text = ["기분이 정말 좋아", "너무 짜증난다"] #예측할 문장들
    #endregion
    
    #region 데이터 준비
    try:
        log("데이터 준비중...")
        from prepare import data
        
        datas = []

        datas.append( data.getMovieReviewData(usingMethod=mode[MovieReviewLoadMode], limits=MovieReviewLimit, max_length=-1, min_length=1) ) ##영화 리뷰 만으론 긍정/부정 예측이 힘듦...
        
        texts, labels = data.merge_tuples(*datas)
        
        log(f"데이터 준비 완료 ({len(texts)}개)")
    except Exception as e:    
        error("데이터 준비", e)
    #endregion
    
    #region 전처리(정규화 등) (토큰화)
    try:
        log("전처리 준비중...")
        import prepare.preprocessing as preprocessing
        
        pre_steps = preprocessing.getNoramlPipeline(using_analyzer=using_analyzer) #pipeline

        log(f"전처리 중... (분석기: {using_analyzer})")
        pre_texts = process_data(texts, pre_steps)

        log("전처리 완료")
    except Exception as e:    
        error("전처리", e)
    #endregion
    
    #region 전처리 데이터 저장 (토큰 전처리)
    if save_preData:
        try:
            log("전처리 데이터 저장중...")
            save(labels, pre_texts, f"{using_analyzer}_preprocessing.txt")
            log("전처리 데이터 저장 완료")
        except Exception as e:    
            error("전처리 데이터 저장", e)
    else:
        log("전처리 데이터 저장 안함")
    #endregion
    
    #region 특징 추출 (벡터화)
    try:
        log(f"특징 추출 준비중... (사용하는 방법: {using_extraction})")
        from prepare import feature_extraction

        extraction_pre_steps = feature_extraction.getNoramlPrePipeline(using_extraction) #pipeline
        extraction_steps = feature_extraction.getNoramlPipeline(using_extraction) #pipeline
        
        log("특징 추출 시작")
        
        if n_gram:
            log("n-gram 적용 중...")
            pre_texts = multi_decorator(feature_extraction.nGram_setting)(pre_texts)

        log("사전 작업 진행 중...")
        extracton_pre_texts = process_data(pre_texts, extraction_pre_steps)
        
        log("작업 중...")
        extracton_texts = process_data(pre_texts, extraction_steps)
        
        log("특징 추출 완료")
    except Exception as e:    
        error("특징 추출", e)
    #endregion
    
    #region 특징 추출 데이터 저장 
    if save_extractionData:
        try:
            log("특징 추출 데이터 저장중...")
            save(labels, extracton_texts, f"{using_analyzer}_{using_extraction}_featureExtraction.txt")
            log("특징 추출 데이터 저장 완료")
        except Exception as e:    
            error("특징 추출 데이터 저장", e)
    else:
        log("특징 추출 데이터 저장 안함")
    #endregion
    
    #region 학습
    try:
        #학습 준비
        log(f"학습 준비중... (랜덤 시드: {random_state})")
        X= toArray(extracton_texts)
        labels = np.array(labels).reshape(len(labels), )
        
        # 학습/테스트 분할
        log(f"데이터 분할 중... (테스트 데이터 비율: {test_size})")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=random_state, stratify=labels) 

        # SVM 모델 학습
        log(f"학습 중... (모델 이름: {using_model})")

        from aiTrain.models import getModels
        model = getModels(using_model) 

        s = time.time()
        model.fit(X_train, y_train)
        
        log(f"학습 완료 (소요 시간: {time.time()-s})")
    except Exception as e:    
        error("학습", e)
    #endregion
    
    #region 모델 저장 
    if save_model:
        try:
            log("모델 저장중...")
            fileName = f"{using_model}_{using_analyzer}_{using_extraction}{"_nGram" if n_gram else ""}" 
            
            path = os.path.join(real_path, "models", fileName)
            import joblib
            joblib.dump(model, ''.join([ path, "_model.pkl" ]))
            joblib.dump(DDD.getDict(), ''.join([ path, "_dict.pkl"]))
            joblib.dump(tfidf.tfidf, ''.join([ path, "_tfidf.pkl"]))
            log("모델 저장 완료")
        except Exception as e:    
            error("모델 저장", e)
    else:
        log("모델 저장 안함")
    #endregion
  
    #region 평가
    try:
        # 평가
        log("평가 중...")
        s = time.time() 
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
        
        print("=== 평가 결과 ===")
        print(report)
        log(f"평가 완료 (소요 시간: {time.time()-s})")
    except Exception as e:    
        error("평가", e)
    #endregion
    
    #region 새 문장 예측
    try:
        # 새 문장 예측을 위한 pipeline
        t = f"새 문장 준비 중... (\"{new_text[0]}\""
        if len(new_text) >= 2:
            t = ''.join( [t, f" ... \"{new_text[-1]}\""] )
            
        t =  ''.join( [t, ")"] )
        log(t)

        #pipeline
        getVec_steps = list(chain.from_iterable(
            [
                pre_steps,
                [multi_decorator(feature_extraction.nGram_setting)] if n_gram else [],
                extraction_steps,
            ]
        )) #전처리 / 벡터화 

        # 새 문장 예측
        log("새 문장 예측 준비 중...")
        new_vec = process_data(new_text, getVec_steps) #벡터로 변환

        log("새 문장 예측 중...")
        new_vec = toArray(new_vec)
        predictions = model.predict(new_vec)

        #예측값 출력
        log("새 문장 예측완료")
        for text, pred in zip(new_text, predictions):
            label = res[int(pred)]
            print(f'"{text}" → {label}')
        
        #사용자 입력 예측하기
        log("사용자 입력 예측 중...")
        while "" != (s := input()):
            new_vec = process_data([s], getVec_steps) #벡터로 변환
            new_vec = toArray(new_vec)

            predictions = model.predict(new_vec)

            label =  res[int(predictions[0])]
            print(f'"{s}" → {label}')
        log("사용자 입력 예측 완료")
    except Exception as e:    
        error("예측", e)
    #endregion

    log("[--종료--]\n")

# 계획
#(완료)1. 데이터를 불러온다 (파일에서)
#(완료)2. 전처리를 한다. (다른 파일에 저장하기)
#(완료)3. 특징추출을 한다(다른 파일에 저장하기) (이 데이터로 학습을 진행)
#(완료)4. 학습 시킨다.
#(완료)5. 테스트 한다.

#TODO
#특징 추출 방법 추가
#ui 만들기
#딥러닝 모델 추가
#특징 추출 최적화 -> mecab? / ...
##중립 상태 판별
#Fuzzy Classification?
#Soft Classification?
#Threshold-based Class Definition?
#Mixture Models?
#Confusion Matrix-based Thresholding?
#Semi-supervised Learning?
#Boundary Class?
#Multiclass Classification?

##가상 환경
#C:\Users\hjcho\Desktop\dream/ai/myenv\Scripts\activate.bat
#deactivate

#python -V
