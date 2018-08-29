import pandas as pandas
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # Read data from file
    csv = 'data/grantnav-src-2018.csv'
    dataframe = pandas.read_csv(csv)

    # Form document from each grant
    dataframe['document'] = dataframe['Title'].map(str) + ' ' + \
                            dataframe['Description'].map(str)

    # Extract features from data set (tokenize, count, weight by TF-IDF)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(dataframe.document)

    # Clustering
    km = KMeans(verbose=False)
    km.fit(X)
    clusters = km.labels_.tolist()

    # Output file with additional column for assigned cluster
    dataframe['cluster'] = clusters
    dataframe.to_csv(path_or_buf='data/grantnav-cluster.csv')
    # print(dataframe.head())

    # Find terms associated with each cluster by applying TF-IDF to a corpus
    # of documents that are the concatenation of all grants assigned to each
    # cluster.
    cluster_docs = dataframe.groupby('cluster')['document'].agg({
        'document': lambda x: ' '.join(x)
    })
    print(cluster_docs.document[1])

    # TODO: fix aggregation format, run TfidfTransformer...