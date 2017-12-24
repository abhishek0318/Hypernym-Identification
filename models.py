"""Contains classifier."""

import sklearn.svm
import numpy as np

class DynamicMarginModel(sklearn.svm.SVC):
    """Wrapper for sklearn.svm.SVC class."""
    def __init__(self, filename1, filename2, *args, **kwargs):
        """Initialses the object.

        Arguments:
            filename1: path to hypernym term embeddings.
            filename2: path to hyponym term embeddings.
        """
        super().__init__(*args, **kwargs)
        self.hypernym_embeddings = {}
        self.hyponym_embeddings = {}

        with open(filename1, 'r') as file:
            for line in file:
                self.hypernym_embeddings[line.split('\t')[0]] = np.array(list(map(float, line.split('\t')[1:])))

        with open(filename2, 'r') as file:
            for line in file:
                self.hyponym_embeddings[line.split('\t')[0]] = np.array(list(map(float, line.split('\t')[1:])))

        self.embedding_size = len(list(self.hypernym_embeddings.values())[0])

    def fit(self, X, *args, **kwargs):
        X = self.word_to_vector(X)
        super().fit(X, *args, **kwargs)

    def predict(self, X):
        X = self.word_to_vector(X)
        return super().predict(X)

    def score(self, X, *args, **kwargs):
        X = self.word_to_vector(X)
        return super().score(X, *args, **kwargs)

    def word_to_vector(self, X):
        """Converts pair of words to concatenation of their embeddings."""

        # If X contains vectors instead of strings, return X as it is.
        if not isinstance(X[0][0], str):
            return X
        
        X_embeddings = []
        for word1, word2 in X:
            embedding1 = self.hypernym_embeddings.get(word1, np.zeros(self.embedding_size))
            embedding2 = self.hyponym_embeddings.get(word2, np.zeros(self.embedding_size))
            norm = [np.linalg.norm(embedding1 - embedding2, ord=1)]
            X_embeddings.append(np.concatenate([embedding1, embedding2, norm]))
        return X_embeddings
