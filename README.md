# Hypernym Identification

This repository contains code for implementation of "Learning Term Embeddings for Hypernymy Identification" [[Yu et al, 2015](https://www.ijcai.org/Proceedings/15/Papers/200.pdf)].

## Overview

Hypernym is defined as a word with a broad meaning constituting a category into which words with more specific meanings fall. For example, animal is a hypernym of dog. Correspondingly, dog is hyponym of animal.


The main idea of this paper is to use word embeddings to represent words and train a SVM classifier on top of the embeddings and L1 norm of the difference between the embeddings.  


Usually word embeddings (like word2vec and Glove) are trained so as to bring highly co-occuring words together. The word embeddings of cat, dog and paws will all be close to each other.  Thus they be can not be used to discriminate between hypernymy and other non-hypernymy relations (like meronyms, cohoponyms). Also as hypernymy relation is non symmetric, one embedding for each word will not suffice. Therefore we need to train two sets of embeddings.  


We train embeddings such that the hypernym embedding of hypernym word becomes close to the hyponym embedding of hyponym. This automatically results in hypernym embedding of cohypernyms and hyponym embedding of hyponyms becoming closer. We train the embeddings in such a way that frequently occurring hypernym hyponym pairs are placed more importance.

## Requirements

* PyTorch 0.3
* NumPy
* scikit-learn

## Usage

> python3 test.py animal dog

## Instructions
* To train embeddings download [Probase dataset](https://concept.research.microsoft.com/Home/Download) and place it in `data` folder as `probase`.

## Datasets used

* [Probase dataset](https://concept.research.microsoft.com/Home/Download) was used for training embeddings.
* [BLESS dataset](https://sites.google.com/site/geometricalmodels/shared-evaluation) was used for training SVM classifier.

## Files description
* `train_embeddings.py` contains the code for training the term embeddings.
* `train_model.py` contains the code for traninig SVM classifier on top of the term embeddings.
* `models.py` contains the classifier class.
* `test.py` is a script that allows to check if two words follow hypernymy relation or not.

