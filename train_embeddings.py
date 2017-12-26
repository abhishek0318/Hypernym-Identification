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
        self.verbose = verbose
        self.data = {}
        self.hypernym_vocab = []
        self.hyponym_vocab = []
        self.net = None
        self.word_hypernym_ix = {}
        self.word_hyponym_ix = {}

    def load_data(self, filename, minimum_count=0, minimum_frequency=0):
        """Load data."""
        with open(filename, 'r') as file:
            for line in file:
                hypernym, hyponym, count = line.split('\t')
                self.hypernym_vocab.append(hypernym)
                self.hyponym_vocab.append(hyponym)
                self.data[(hypernym, hyponym)] = int(count)

        if self.verbose > 1:
            print("File loaded.")

        self.filter_data(minimum_count, minimum_frequency)

        if self.verbose > 1:
            print("Data filtered.")

        # Remove duplicate words
        self.hypernym_vocab = tuple(set(self.hypernym_vocab))
        self.hyponym_vocab = tuple(set(self.hyponym_vocab))

        if self.verbose > 1:
            print("Duplicate words removed.")

        # Dictionary which maps word to its index.
        self.word_hypernym_ix = {word: i for i, word in enumerate(self.hypernym_vocab)}
        self.word_hyponym_ix = {word: i for i, word in enumerate(self.hyponym_vocab)}

        self.net = EmbeddingTrainer.Net(len(self.hypernym_vocab), len(self.hyponym_vocab), self.embedding_size)

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

    class Batch():
        """Used for temporarily storing data before making gradient update"""
        def __init__(self):
            self.hypernyms = []
            self.hyponyms = []
            self.hypernyms1 = []
            self.hyponyms1 = []
            self.counts = []
            self.counts1 = []

        def add(self, hypernym, hyponym, count, hypernym1, hyponym1, count1):
            """Add example to batch."""
            self.hypernyms.append(hypernym)
            self.hyponyms.append(hyponym)
            self.hypernyms1.append(hypernym1)
            self.hyponyms1.append(hyponym1)
            self.counts.append(count)
            self.counts1.append(count1)

        def is_empty(self):
            """Return true if the batch is empty"""
            return len(self.hypernyms) == 0

    def train(self, epochs=10, batch_size=32, lr=0.01, gpu=False, save_location=None):
        """Train the network"""

        if gpu:
            self.net.cuda()

        optimizer = optim.SGD(self.net.parameters(), lr=lr)

        # Used for temporarily storing data before gradient is updated
        batch = EmbeddingTrainer.Batch()

        for epoch in range(epochs):
            # Counts number of examples encountered during an eopch
            counter = 0
            cost_epoch = 0

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
                    batch.add(hypernym, hyponym, count, hypernym1, hyponym1, count1)
                    counter += 1

                    # Update the gradients after every batch_size number of iterations
                    if counter % batch_size == 0:
                        # Get index of words
                        hypernyms_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym] for hypernym in batch.hypernyms]))
                        hyponyms_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym] for hyponym in batch.hyponyms]))

                        hypernyms1_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym1] for hypernym1 in batch.hypernyms1]))
                        hyponyms1_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym1] for hyponym1 in batch.hyponyms1]))

                        counts_var = Variable(torch.Tensor(batch.counts))
                        counts1_var = Variable(torch.Tensor(batch.counts1))

                        # Compute cost
                        if gpu:
                            cost = self.net(hypernyms_ix.cuda(), hyponyms_ix.cuda(), counts_var.cuda(),\
                                            hypernyms1_ix.cuda(), hyponyms1_ix.cuda(), counts1_var.cuda())
                        else:                        
                            cost = self.net(hypernyms_ix, hyponyms_ix, counts_var, hypernyms1_ix, hyponyms1_ix, counts1_var)
                            

                        cost_epoch += torch.sum(cost).data[0]
                        optimizer.zero_grad()
                        if gpu:
                            cost.backward(torch.ones(cost.size()).cuda())
                        else:
                            cost.backward(torch.ones(cost.size()))
                        optimizer.step()

                        # Reset temporarily storage
                        batch.__init__()

                    if counter % 1000 == 0:
                        if self.verbose > 1:
                            print("Epoch {} of {}: {} of {} ({:.4f}%)".format(epoch + 1, epochs,
                                                                              counter, self.total_examples_epoch,
                                                                              (counter / self.total_examples_epoch) * 100))

            # Update the gradients for remaining data
            if not batch.is_empty():
                hypernyms_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym] for hypernym in batch.hypernyms]))
                hyponyms_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym] for hyponym in batch.hyponyms]))

                hypernyms1_ix = Variable(torch.LongTensor([self.word_hypernym_ix[hypernym1] for hypernym1 in batch.hypernyms1]))
                hyponyms1_ix = Variable(torch.LongTensor([self.word_hyponym_ix[hyponym1] for hyponym1 in batch.hyponyms1]))

                counts_var = Variable(torch.Tensor(batch.counts))
                counts1_var = Variable(torch.Tensor(batch.counts1))

                # Compute cost
                if gpu:
                    cost = self.net(hypernyms_ix.cuda(), hyponyms_ix.cuda(), counts_var.cuda(),\
                                    hypernyms1_ix.cuda(), hyponyms1_ix.cuda(), counts1_var.cuda())
                else:                        
                    cost = self.net(hypernyms_ix, hyponyms_ix, counts_var, hypernyms1_ix, hyponyms1_ix, counts1_var)

                cost_epoch += torch.sum(cost).data[0]
                optimizer.zero_grad()
                if gpu:
                    cost.backward(torch.ones(cost.size()).cuda())
                else:
                    cost.backward(torch.ones(cost.size()))
                optimizer.step()

                # Reset temporarily storage
                batch.__init__()

            if self.verbose:
                print("Average cost in epoch {}: {}".format(epoch + 1, cost_epoch / counter))

            if save_location:
                self.save_embeddings(*save_location)

    def save_embeddings(self, filename1, filename2):
        """Save the embeddings to a file"""

        weights = list(self.net.parameters())
        if weights[0].data.is_cuda:
            hypernym_weights = weights[0].data.cpu().numpy()
            hyponym_weights = weights[1].data.cpu().numpy()
        else:
            hypernym_weights = weights[0].data.numpy()
            hyponym_weights = weights[1].data.numpy()

        with open(filename1, 'w') as file:
            for hypernym, hypernym_weight in zip(self.hypernym_vocab, hypernym_weights):
                file.write(hypernym + "\t" + self.array_to_string(hypernym_weight) + "\n")

        with open(filename2, 'w') as file:
            for hyponym, hyponym_weight in zip(self.hyponym_vocab, hyponym_weights):
                file.write(hyponym + "\t" + self.array_to_string(hyponym_weight) + "\n")

        if self.verbose > 1:
            print("Embeddings saved.")

    def array_to_string(self, array):
        """Convert numpy array to string"""
        string = '\t'.join(map(str, array))
        return string

    def filter_data(self, minimum_count, minimum_frequency):
        """Filter data.

        Removes data with count less than minimum count and terms with frequency less than minimum_frequency
        """
        # Counts number of times a word has appeared in the data
        counter = Counter(self.hypernym_vocab)
        counter.update(self.hyponym_vocab)

        self.hypernym_vocab = []
        self.hyponym_vocab = []

        filtered_data = {}
        self.total_examples_epoch = 0
        for (hypernym, hyponym), count in self.data.items():
            if counter[hypernym] > minimum_frequency and counter[hyponym] > minimum_frequency and count > minimum_count:
                self.total_examples_epoch += count
                filtered_data[(hypernym, hyponym)] = count
                self.hypernym_vocab.append(hypernym)
                self.hyponym_vocab.append(hyponym)

        self.data = filtered_data

if __name__ == "__main__":
    trainer = EmbeddingTrainer(embedding_size=50, verbose=2)
    trainer.load_data(os.path.join('data', 'probase'), minimum_count=5, minimum_frequency=10)
    save_location = (os.path.join('data', 'hypernym_embedding'), os.path.join('data', 'hyponym_embedding'))
    trainer.train(epochs=20, batch_size=32, lr=0.01, gpu=False, save_location=save_location)