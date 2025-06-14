if __name__ == "__main__":
    from utils import *

    using_model = "SGDClassifier"#input("using_model: ")
    using_analyzer = "Komoran"#input("using_analyzer: ")
    using_extraction = "TF-IDF"#input("using_extraction: ")
    n_gram = True #if input("n-gram(y/n): ") == "y" else False

    fileName = f"{using_model}_{using_analyzer}_{using_extraction}{"_nGram" if n_gram else ""}"
    print(fileName)
    
    path = os.path.join(real_path, "models", fileName)
    
    import joblib
    loadedModel = joblib.load(''.join([path, "_model.pkl"]))
    DDD.setDict(joblib.load(''.join([path, "_dict.pkl"])))
    tfidf.set(joblib.load(''.join([path, "_tfidf.pkl"])))
    
    from main_func import multi_decorator, process_data, res
    import preprocessing, feature_extraction

    pre_steps = preprocessing.getNoramlPipeline(using_analyzer=using_analyzer) #pipeline
    extraction_steps = feature_extraction.getNoramlPipeline(using_extraction) #pipeline
    
    #pipeline
    getVec_steps = list(chain.from_iterable(
        [
            pre_steps,
            [multi_decorator(feature_extraction.nGram_setting)] if n_gram else [],
            extraction_steps,
        ]
    )) #전처리 / 벡터화

    while "" != (s := input()):
        new_vec = process_data([s], getVec_steps) #벡터로 변환
        new_vec = toArray(new_vec)

        predictions = loadedModel.predict(new_vec)

        label =  res[int(predictions[0])]
        print(f'"{s}" → {label}')
