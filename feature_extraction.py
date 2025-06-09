from utils import *
#region Bag of Words (BoW, 단어 빈도 기반)
def vocabularyCreate(tokenized_document):
    global vocabulary
    vocabulary = set(vocabulary)
    for doc in tokenized_document:
        vocabulary.add(doc)
    vocabulary = tuple(vocabulary)
    return tokenized_document

def document_to_vector(doc):
    global vocabulary
    vector = [0] * len(vocabulary)
    for word in doc:
        if word in vocabulary:
            index = vocabulary.index(word)
            vector[index] += 1
    return vector
#endregion
#region n-gram
def nGram_setting(tokenized_document):
    max_n = 2
    ngram_tokens = []
    for n in range(1, max_n+1):
        if len(tokenized_document) >= n:
            ngram_tokens += [''.join(tokenized_document[i:i+n]) for i in range(len(tokenized_document)-n+1)]
    return ngram_tokens
#endregion
#TF-IDF (단어 빈도 vs 문서 중요도)
def TF_IDF_init(bow_matrix):
    return tfidf.fit_transform(bow_matrix)
#Word2Vec / FastText (단어 임베딩)
#BERT 계열 임베딩 (Contextual Embedding)
