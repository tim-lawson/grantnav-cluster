import pandas as pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

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
kmeans = KMeans(verbose=True)
kmeans.fit(X)
print(metrics.silhouette_score(X, kmeans.labels_))