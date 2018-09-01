import itertools
import numpy as np
import pandas as pandas

from time import time
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    # Read data from file
    t0 = time()
    csv = 'data/grantnav-src-2018.csv'
    dataframe = pandas.read_csv(csv)
    print('Read data from file in {:.6f}s'.format(time() - t0))

    # Form document from each grant
    t0 = time()
    dataframe['document'] = dataframe['Title'].map(str) + ' ' + \
                            dataframe['Description'].map(str)

    # Form Tf-idf-weighted document-term matrix from corpus
    tfidf_vectorizer = TfidfVectorizer()
    # This is a sparse matrix of Tf-idf-weightings indexed by (document, term)
    doc_term_matrix = tfidf_vectorizer.fit_transform(dataframe['document'])
    print('Formed Tf-idf-weighted document-term matrix in {:.6f}s'
          .format(time() - t0))

    # Read pre-trained word embeddings from file
    t0 = time()
    with open('data\glove.6B.50d.txt', 'rb') as lines:
        # Note that map returns an iterator in Python 3
        word2vec = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
                    for line in lines}
        dim = len(word2vec[b'the'])

    print('Read pre-trained word embeddings from file in {:.6f}s'
          .format(time() - t0))

    # Form mapping of feature indices to embeddings
    t0 = time()
    embed = {v: word2vec.get(k.encode('UTF-8'), np.zeros(dim))
             for k, v in tfidf_vectorizer.vocabulary_.items()}
    print('Formed mapping of feature indices to embeddings in {:.6f}s'
          .format(time() - t0))

    # Form document embeddings from average of Tf-idf-weighted word embeddings
    t0 = time()
    num_docs, num_terms = doc_term_matrix.shape
    doc_embed_matrix = np.zeros((num_docs, dim))
    doc_num_terms = np.zeros(num_docs)

    docs, terms = doc_term_matrix.nonzero()
    for doc, term in zip(docs, terms):
        # Inefficient way of calculating the average...
        doc_num_terms[doc] = doc_num_terms[doc] + 1
        doc_embed_matrix[doc, :] = ((doc_embed_matrix[doc, :] * \
                                     doc_num_terms[doc] - 1) + \
                                    (doc_term_matrix[doc, term] * embed[term])
                                   ) / doc_num_terms[doc]

    print('Formed document embedding matrix in {:.6f}s'
          .format(time() - t0))
