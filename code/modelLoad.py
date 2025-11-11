from func.utils import *

using_model = "SGDClassifier"#input("using_model: ")
using_analyzer = "Komoran"#input("using_analyzer: ")
using_extraction = "TF-IDF"#input("using_extraction: ")
n_gram = True #if input("n-gram(y/n): ") == "y" else False

def file_name():
    fileName = f"{using_model}_{using_analyzer}_{using_extraction}{"_nGram" if n_gram else ""}"

    return fileName

def modelLoad(path):
    import joblib
    global loadedModel, DDD, tfidf

    loadedModel = joblib.load(''.join([path, "_model.pkl"]))
    DDD.setDict(joblib.load(''.join([path, "_dict.pkl"])))
    tfidf.set(joblib.load(''.join([path, "_tfidf.pkl"])))

    return loadedModel

def preparePre():
    from prepare import preprocessing
    pre_steps = preprocessing.getNoramlPipeline(using_analyzer=using_analyzer) #pipeline
    from prepare import feature_extraction
    extraction_steps = feature_extraction.getNoramlPipeline(using_extraction) #pipeline

    #pipeline
    getVec_steps = list(chain.from_iterable(
        [
            pre_steps,
            [feature_extraction.nGram_setting_mul] if n_gram else [],
            extraction_steps,
        ]
    )) #전처리 / 벡터화

    return getVec_steps

if __name__ == "__main__":
    from func.main_func import process_data, res
    from func.main_func import log, error
    log("[--모델 불러오기--]")

    log("모델 준비중")
    fileName= file_name()
    log(f"파일 이름: {fileName}")
    
    log("모델 로딩중")
    path = os.path.join(real_path, "models", fileName)
    modelLoad(path=path)


    log("전처리 및 특징 추출 준비")
    getVec_steps= preparePre()
    
    log("시작")
    try:
        while "" != (s := input()):
            new_vec = process_data([s], getVec_steps) #벡터로 변환
            new_vec = toArray(new_vec)

            predictions = loadedModel.predict(new_vec)

            label =  res[int(predictions[0])]
            print(f'"{s}" → {label}')
    except Exception as e:    
        error("모델 테스트", e)
    log("[--종료--]\n")
