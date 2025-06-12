if __name__ == "__main__":
    import utils
    from main_func import multi_decorator, process_data, res
    import preprocessing, feature_extraction

    using_model = "SGDClassifier"#input("using_model: ")
    using_analyzer = "Komoran"#input("using_analyzer: ")
    using_extraction = "TF-IDF"#input("using_extraction: ")
    n_gram = True #if input("n-gram(y/n): ") == "y" else False

    fileName = f"{using_model}_{using_analyzer}_{using_extraction}{"_nGram" if n_gram else ""}"
    print(fileName)
    
    path = utils.os.path.join(utils.real_path, "models", fileName)

    loadedModel = utils.joblib.load(path+"_model.pkl")
    utils.DDD.setDict(utils.joblib.load(path+"_dict.pkl"))
    utils.tfidf.set(utils.joblib.load(path+"_tfidf.pkl"))

    pre_steps = preprocessing.getNoramlPipeline(using_analyzer=using_analyzer) #pipeline
    extraction_steps = feature_extraction.getNoramlPipeline(using_extraction) #pipeline

    #pipeline
    getVec_steps = \
        pre_steps + \
            ([multi_decorator(feature_extraction.nGram_setting)] if n_gram else []) \
            + extraction_steps #전처리 + 벡터화 

    while "" != (s := input()):
        new_vec = process_data([s], getVec_steps) #벡터로 변환
        new_vec = utils.toArray(new_vec)

        predictions = loadedModel.predict(new_vec)

        label =  res[predictions[0]]
        print(f'"{s}" → {label}')
