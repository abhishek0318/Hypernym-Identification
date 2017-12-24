"""Use BLESS dataset to train classifier and save it to disk."""

import os.path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from models import DynamicMarginModel

def process_bless(filename):
    """Reads in BLESS.txt and returns word pairs."""
    hypernym_pairs = []
    cohypernym_pairs = []
    meronym_pairs = []
    random_pairs = []

    with open(filename, 'r') as file:
        for line in file:
            concept, _, relation, relatum = line.split()
            if relation == 'coord':
                cohypernym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'hyper':
                hypernym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'mero':
                meronym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'random-n':
                random_pairs.append((relatum[:-2], concept[:-2]))

    return hypernym_pairs, cohypernym_pairs, meronym_pairs, random_pairs

if __name__ == "__main__":
    hypernym_pairs, cohypernym_pairs, meronym_pairs, random_pairs = process_bless(os.path.join('data', 'BLESS.txt'))

    X = []
    X.extend(hypernym_pairs)
    X.extend(cohypernym_pairs)
    X.extend(meronym_pairs)
    X.extend(random_pairs)

    y = []
    y.extend([1 for _ in range(len(hypernym_pairs))])
    y.extend([0 for _ in range(len(cohypernym_pairs))])
    y.extend([0 for _ in range(len(meronym_pairs))])
    y.extend([0 for _ in range(len(random_pairs))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    model = DynamicMarginModel(os.path.join('data', 'hypernym_embedding'),\
                 os.path.join('data', 'hyponym_embedding'), C=8, class_weight='balanced')

    model.fit(X_train, y_train)
    print('Train score: {}'.format(model.score(X_train, y_train)))
    print('Test score: {}'.format(model.score(X_test, y_test)))
    print(classification_report(y_test, model.predict(X_test)))

    model.fit(X, y)
    joblib.dump(model, os.path.join('data', 'trained_model.pkl'))
