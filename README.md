# Text clustering for grant categorization

## Notes
If there is no embedding for a word in the corpus, i.e. the word occurs in the corpus we are looking at but not in the corpus used to generate the embeddings, then it is ignored. We might alternatively use an array of zeros or a unit vector with random direction, but neither contributes to the meaning of the subsequent document embedding. If there is no embedding for any of the words in a document, an array of zeros is taken as its embedding.

I suspect it would be simpler to implement the calculation of the embeddings as a sklearn estimator, but I'm not confident enough to do so at the moment. (It would be initialised with a path to the GloVe-format txt file and the fit method would include building the dict of embeddings.)

## TODO
* Apply spherical K-means to document embeddings. Read about von Mises Fisher distribution and especially soft clustering
* How to evaluate clusters visually (for intuition) and numerically (for hyperparameter search)?
* Interpretable clusters: save most common terms relative to other clusters (TF-IDF), or the closest word embeddings?
* Implement whole model as sklearn pipeline - search over number of clusters...
* Add command line arguments for path to input data...
* Investigate how the model scales with number of grants, number of clusters...