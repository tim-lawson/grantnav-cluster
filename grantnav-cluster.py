import itertools
import numpy as np
import pandas as pandas

from time import time
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
# from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    # Read data from file
    csv = 'data/grantnav-20180904133708.csv'
    t0 = time()
    dataframe = pandas.read_csv(csv)
    print('Read data from file in {:.6f}s'.format(time() - t0))

    # Form document from each grant
    dataframe['document'] = dataframe['Title'].map(str) + ' ' + \
                            dataframe['Description'].map(str)

    # Form Tf-idf-weighted document-term matrix from corpus
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    t0 = time()
    # This is a sparse matrix of Tf-idf-weightings indexed by (document, term)
    doc_term_matrix = tfidf_vectorizer.fit_transform(dataframe['document'])
    print('Formed Tf-idf-weighted document-term matrix in {:.6f}s'
          .format(time() - t0))

    # Read pre-trained word embeddings from file
    t0 = time()
    with open('data\glove.6B.100d.txt', 'rb') as lines:
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
    num_docs, num_terms = doc_term_matrix.shape
    doc_embed_matrix = np.zeros((num_docs, dim))
    doc_num_terms = np.zeros(num_docs)

    t0 = time()
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

    # Dimensionality reduction (LSA)
    t0 = time()
    svd = TruncatedSVD(n_components=2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    lsa_embed_matrix = lsa.fit_transform(doc_embed_matrix)
    print('Truncated SVD in {:.6f}s, explained variance {:.3f}'
          .format(time() - t0), svd.explained_variance_ratio_.sum())

    # Cluster document embeddings
    kmeans = KMeans(n_clusters=10)
    # skm = SphericalKMeans(n_clusters=10)
    t0 = time()
    kmeans.fit(lsa_embed_matrix)
    # skm.fit(doc_embed_matrix)
    print('Clustered document embeddings in {:.6f}s'
          .format(time() - t0))

    # Visualise clusters in 2-d
    # pca = PCA(n_components=2).fit_transform(doc_embed_matrix)
    # tsne = TSNE(n_components=2, verbose=True).fit_transform(doc_embed_matrix)

    plt.figure()
    plt.scatter(lsa_embed_matrix[:,0], lsa_embed_matrix[:,1])
    # plt.scatter(pca[:,0], pca[:,1], c=kmeans.labels_.astype(float))
    # plt.scatter(tsne[:,0], tsne[:,1], c=kmeans.labels_.astype(float))
    # plt.scatter(tsne[:,0], tsne[:,1], c=skm.labels_.astype(float))
    plt.show()