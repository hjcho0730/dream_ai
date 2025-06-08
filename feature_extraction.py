from utils import *
#Bag of Words (BoW, 단어 빈도 기반)

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

#TF-IDF (단어 빈도 vs 문서 중요도)
#Word2Vec / FastText (단어 임베딩)
#BERT 계열 임베딩 (Contextual Embedding)
