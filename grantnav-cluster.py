import itertools
import numpy as np
import pandas as pd

from time import time
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from spherecluster import SphericalKMeans

# https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices
def top_n(row_data, row_indices, n):
    i = row_data.argsort()[-n:]
    top_values = row_data[i]
    top_indices = row_indices[i]

    return top_values, top_indices, i

if __name__ == "__main__":
    # Arguments
    input_data = 'data/grantnav-src-2018.csv'
    output_data = 'data/grantnav-cluster.csv'
    output_clusters = 'data/clusters.csv'
    n_clusters = 10
    n_terms = 10

    # Read data from file
    csv = 'data/grantnav-20180904133708.csv'
    t0 = time()
    grants = pd.read_csv(input_data)
    print('Read data from file in {:.6f}s'.format(time() - t0))

    # Form document from each grant
    t0 = time()
    grants['document'] = grants['Title'].map(str) + ' ' + \
                         grants['Description'].map(str)

    # Form Tf-idf-weighted document-term matrix from corpus
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # This is a sparse matrix of Tf-idf-weightings indexed by (document, term)
    doc_term_matrix = tfidf_vectorizer.fit_transform(grants['document'])
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

    # Cluster documents by spherical K-means (cluster centroids are projected
    # onto the unit hypersphere)
    t0 = time()
    skm = SphericalKMeans(n_clusters=n_clusters)
    skm.fit(doc_embed_matrix)
    
    print('Clustered documents by spherical K-means in {:.6f}s'
          .format(time() - t0))

    # Add column to dataframe to hold assigned cluster and output to CSV
    grants['cluster'] = skm.labels_.tolist()
    grants.to_csv(path_or_buf=output_data)
    
    # Find terms associated with each cluster by applying Tf-idf to a corpus of
    # documents that are the concatenation of all grants assigned to each cluster.
    t0 = time()
    cluster_docs = grants.groupby('cluster')['document'].agg(lambda x: ' '.join(x))
    
    cluster_vectorizer = TfidfVectorizer(stop_words='english')
    cluster_vectorizer.fit(cluster_docs)

    cluster_term_matrix = cluster_vectorizer.transform(cluster_docs)
    
    print('Formed Tf-idf-weighted cluster-term matrix in {:.6f}s'
          .format(time() - t0))

    # Iterate document-term matrix to retrieve top N terms per cluster
    t0 = time()
    features = cluster_vectorizer.get_feature_names()

    cluster_term_matrix_ll = cluster_term_matrix.tolil()
    cluster_top_n_terms = []
    for i in range(cluster_term_matrix_ll.shape[0]):
        data, rows = top_n(np.array(cluster_term_matrix_ll.data[i]),
                           np.array(cluster_term_matrix_ll.rows[i]),
                           n_terms)[:2]
        
        for feature, weight in zip(rows, data):
            cluster_top_n_terms.append({
                'cluster': i,
                'feature': feature,
                'word': features[feature],
                'weight': weight
            })

    print('Retrieved top {} terms per cluster in {:.6f}s'
          .format(n_terms, time() - t0))

    clusters = pd.DataFrame(cluster_top_n_terms)
    clusters.to_csv(path_or_buf=output_clusters)
