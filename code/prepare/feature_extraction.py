from func.utils import *
from func.main_func import multi_decorator

#region Bag of Words (BoW, 단어 빈도 기반)
def vocabularyCreate(tokenized_document):
    global vocabulary
    for doc in tokenized_document:
        vocabulary.add(doc)
    return tokenized_document

@multi_decorator
def vocabularyCreate_mul(tokenized_document):
    global vocabulary
    for doc in tokenized_document:
        vocabulary.add(doc)
    return tokenized_document

def vocabularyToDict(doc):
    global vocabulary, vector
    dd = {}
    for idx, i in enumerate(vocabulary):
        dd[i] = idx
    DDD.setDict(dd)
    return doc

def deleteTopN(doc, n=50):
    data = sorted(DDD.getDict().items(), reverse=True, key=lambda x: x[1])[:n]
    for i in range(min(n, len(data))):
        DDD.deleteVal(data[i][0])
    return doc
def document_to_vector(docs):
    vocabularyDict = DDD.getDict()
    vector = lil_matrix((len(docs), len(vocabularyDict)))
    for i in range(len(docs)):
        for word in docs[i]:
            if word in vocabularyDict:
                index = vocabularyDict[word]#vocabulary.index(word)
                vector[i, index] += 1
    return vector.tocsr()
#endregion
#region n-gram
def nGram_setting(tokenized_document):
    max_n = 2
    ngram_tokens = []
    for n in range(1, max_n+1):
        if len(tokenized_document) >= n:
            ngram_tokens.extend(''.join(tokenized_document[i:i+n]) for i in range(len(tokenized_document)-n+1))
    return ngram_tokens

@multi_decorator
def nGram_setting_mul(tokenized_document):
    max_n = 2
    ngram_tokens = []
    for n in range(1, max_n+1):
        if len(tokenized_document) >= n:
            ngram_tokens.extend(''.join(tokenized_document[i:i+n]) for i in range(len(tokenized_document)-n+1))
    return ngram_tokens
#endregion
#TF-IDF (단어 빈도 vs 문서 중요도)
def TF_IDF_init(bow_matrix):
    tfidf.tfidf.fit(bow_matrix)
    return bow_matrix

def prune_low_tfidf(tfidf_matrix: csr_matrix, threshold: float = 0.005) -> csr_matrix:
    """
    TF-IDF 희소 행렬에서 중요도가 낮은 값(= threshold 이하)을 제거
    Parameters:
        tfidf_matrix: scipy.sparse.csr_matrix
        threshold: 이 값 이하의 TF-IDF는 0으로 처리
    Returns:
        새로운 csr_matrix
    """
    tfidf_matrix = tfidf_matrix.tocsr()  # 보장
    tfidf_matrix.data[tfidf_matrix.data < threshold] = 0.0
    tfidf_matrix.eliminate_zeros()  # 0이 된 값들 제거
    return tfidf_matrix

#Word2Vec / FastText (단어 임베딩)
#BERT 계열 임베딩 (Contextual Embedding)

def getNoramlPrePipeline(using_extraction):

    extraction_List = ["BoW", "TF-IDF"]
    if using_extraction not in extraction_List:
        return []

    extraction_pre_steps = []
    if using_extraction == "BoW":
        extraction_pre_steps = [
            vocabularyCreate_mul, #사전 제작
            vocabularyToDict,
            deleteTopN,
        ]
    elif using_extraction == "TF-IDF":
        extraction_pre_steps = [
            vocabularyCreate_mul,
            vocabularyToDict,
            deleteTopN,
            document_to_vector,
            TF_IDF_init,
        ]

    return extraction_pre_steps

def getNoramlPipeline(using_extraction):
    extraction_List = ["BoW", "TF-IDF"]
    if using_extraction not in extraction_List:
        return []

    extraction_steps = []
    if using_extraction == "BoW":
        extraction_steps = [
            document_to_vector,
            #tfidf.tfidf.transform,
        ]
    elif using_extraction == "TF-IDF":
        extraction_steps = [
            document_to_vector,
            tfidf.tfidf.transform,
            prune_low_tfidf,
        ]

    return extraction_steps
