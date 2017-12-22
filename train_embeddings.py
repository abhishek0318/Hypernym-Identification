"""Contains class for training embeddings"""

from collections import Counter
import os.path
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

class EmbeddingTrainer():
    """Contains methods for loading, training and saving embeddings."""
    def __init__(self, embedding_size, verbose=0):
        self.embedding_size = embedding_size
        self.data = {}
        self.hypernym_vocab = []
        self.hyponym_vocab = []
        self.hypernym_vocabulary_size = 0
        self.hyponym_vocabulary_size = 0
        self.word_hypernym_ix = {}
        self.word_hyponym_ix = {}
        self.verbose = verbose
    
    def load_data(self, filename):
        """Load data."""
        with open(filename, 'r') as file:
            for line in file:
                hypernym, hyponym, count = line.split('\t')
                self.hypernym_vocab.append(hypernym)
                self.hyponym_vocab.append(hyponym)
                self.data[(hypernym, hyponym)] = int(count)

        if self.verbose > 1:
            print("File loaded.")

        self.filter_data(minimum_count=5, minimum_frequency=0)

        if self.verbose > 1:
            print("Data filtered.")
        
        # Remove duplicate words
        self.hypernym_vocab = set(self.hypernym_vocab)
        self.hyponym_vocab = set(self.hyponym_vocab)

        self.hypernym_vocab = tuple(self.hypernym_vocab)
        self.hyponym_vocab = tuple(self.hyponym_vocab)

        if self.verbose > 1:
            print("Duplicate words removed.")

        self.hypernym_vocabulary_size = len(self.hypernym_vocab)
        self.hyponym_vocabulary_size = len(self.hyponym_vocab)

        # Dictionary which maps word to its index.
        self.word_hypernym_ix = {word: i for i, word in enumerate(self.hypernym_vocab)}
        self.word_hyponym_ix = {word: i for i, word in enumerate(self.hyponym_vocab)}

        if self.verbose > 1:
            print("Index dictionary created.")

    class Net(nn.Module):
        """Network for training term embeddings."""
        def __init__(self, hypernym_vocabulary_size, hyponym_vocabulary_size, embedding_size):
            super(EmbeddingTrainer.Net, self).__init__()

            self.hypernym_vocabulary_size = hypernym_vocabulary_size
            self.hyponym_vocabulary_size = hyponym_vocabulary_size
            self.embedding_size = embedding_size

            self.hypernym_embedding = nn.Embedding(self.hypernym_vocabulary_size, self.embedding_size)
            self.hyponym_embedding = nn.Embedding(self.hyponym_vocabulary_size, self.embedding_size)

        def forward(self, hypernym_ix, hyponym_ix, count, hypernym_ix1, hyponym_ix1, count1):
            embed_hypernym = self.hypernym_embedding(hypernym_ix)
            embed_hyponym = self.hyponym_embedding(hyponym_ix)

            embed_hypernym1 = self.hypernym_embedding(hypernym_ix1)
            embed_hyponym1 = self.hyponym_embedding(hyponym_ix1)

            norm = (embed_hypernym - embed_hyponym).norm(p=1, dim=1)
            norm1 = (embed_hypernym1 - embed_hyponym1).norm(p=1, dim=1)

            cost = (norm + count.log1p()) - (norm1 + count1.log1p())
            cost = torch.clamp(cost, min=0)
            return cost

    def train(self, epochs=10, batch_size=32):
        """Train the network"""

        net = EmbeddingTrainer.Net(self.hypernym_vocabulary_size, self.hyponym_vocabulary_size, self.embedding_size)
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        # Used for temporarily storing data before gradient is updated
        hypernyms = []
        hyponyms = []
        hypernyms1 = []
        hyponyms1 = []
        counts = []
        counts1 = []

        for epoch in range(epochs):
            # Counts number of examples trained on during an eopch
            counter = 0 
            cost_in_epoch = 0

            for (hypernym, hyponym), count in self.data.items():
                for _ in range(count):
                    # Randomly change hypernym or hyponym
                    choice = random.choice([0, 1])
                    if choice is 0:
                        hypernym1 = random.choice(self.hypernym_vocab)
                        hyponym1 = hyponym
                    else:
                        hypernym1 = hypernym
                        hyponym1 = random.choice(self.hyponym_vocab)

                    count1 = self.data.get((hypernym1, hyponym1), 0)               

                    hypernyms.append(hypernym)
                    hyponyms.append(hyponym)
                    hypernyms1.append(hypernym1)
                    hyponyms1.append(hyponym1)
                    counts.append(count)
                    counts1.append(count1)

                    counter += 1

                    # Update the gradients after every batch_size number of iterations
                    if counter % batch_size == 0:
                        # Get index of words
                        hypernyms_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym] for hypernym in hypernyms]))
                        hyponyms_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym] for hyponym in hyponyms]))

                        hypernyms1_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym1] for hypernym1 in hypernyms1]))
                        hyponyms1_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym1] for hyponym1 in hyponyms1]))

                        # Compute cost
                        cost = net(hypernyms_ix, hyponyms_ix, Variable(torch.Tensor([count])),\
                                    hypernyms1_ix, hyponyms1_ix, Variable(torch.Tensor([count1])))

                        cost_in_epoch += torch.sum(cost).data[0]
                        optimizer.zero_grad()
                        cost.backward(torch.ones(batch_size))
                        optimizer.step()

                        # Reset temporarily storage
                        hypernyms = []
                        hyponyms = []
                        hypernyms1 = []
                        hyponyms1 = []
                        counts = []
                        counts1 = []

                    if counter % 1000 == 0:
                        if self.verbose > 1:
                            print("Weights trained over {} examples in Epoch {}.".format(counter, epoch + 1))

            # Update the gradients for remaining data
            if len(hypernyms) != 0:
                hypernyms_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym] for hypernym in hypernyms]))
                hyponyms_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym] for hyponym in hyponyms]))

                hypernyms1_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym1] for hypernym1 in hypernyms1]))
                hyponyms1_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym1] for hyponym1 in hyponyms1]))

                cost = net(hypernyms_ix, hyponyms_ix, Variable(torch.Tensor([count])),\
                            hypernyms1_ix, hyponyms1_ix, Variable(torch.Tensor([count1])))

                cost_in_epoch += torch.sum(cost).data[0]
                optimizer.zero_grad()
                cost.backward(torch.ones(cost.size()))
                optimizer.step()

                # Reset temporarily storage
                hypernyms = []
                hyponyms = []
                hypernyms1 = []
                hyponyms1 = []
                counts = []
                counts1 = []

            if self.verbose:
                print("Train epoch {}: {}".format(epoch + 1, cost_in_epoch / counter))

        weights = list(net.parameters())
        self.hypernym_weights = weights[0].data.numpy()
        self.hyponym_weights = weights[1].data.numpy()

    def save_embeddings(self, filename1, filename2):
        """Save the embeddings to a file"""
        with open(filename1, 'w') as file:
            for hypernym, hypernym_weight in zip(self.hypernym_vocab, self.hypernym_weights):
                file.write(hypernym + " " + self.array_to_string(hypernym_weight) + "\n")

        with open(filename2, 'w') as file:
            for hyponym, hyponym_weight in zip(self.hyponym_vocab, self.hyponym_weights):
                file.write(hyponym + " " + self.array_to_string(hyponym_weight) + "\n")

        if self.verbose > 1:
            "Embeddigs saved."

    def array_to_string(self, array):
        """Convert numpy array to string"""
        string = ' '.join(map(lambda x: str(x), array))
        return string

    def filter_data(self, minimum_count, minimum_frequency):
        counter = Counter(self.hypernym_vocab)
        counter.update(self.hyponym_vocab)

        self.hypernym_vocab = []
        self.hyponym_vocab = []

        filtered_data = {}
        for (hypernym, hyponym), count in self.data.items():
            if counter[hypernym] > minimum_frequency and counter[hyponym] > minimum_frequency and count > minimum_count:
                filtered_data[(hypernym, hyponym)] = count
                self.hypernym_vocab.append(hypernym)
                self.hyponym_vocab.append(hyponym)

        self.data = filtered_data

if __name__ == "__main__":
    trainer = EmbeddingTrainer(embedding_size=50, verbose=1)
    trainer.load_data(os.path.join('data', 'sample_data3'))
    trainer.train(epochs=30, batch_size=32)
    trainer.save_embeddings(os.path.join('data', 'hypernym_embedding'),\
                             os.path.join('data', 'hyponym_embedding'))