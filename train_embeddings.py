"""Contains class for training embeddings"""

import os.path
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

class EmbeddingTrainer():
    """Contains methods for loading, training and saving embeddings."""
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.data = {}
        self.hypernym_vocab = []
        self.hyponym_vocab = []
        self.hypernym_vocabulary_size = 0
        self.hyponym_vocabulary_size = 0
        self.word_hypernym_ix = {}
        self.word_hyponym_ix = {}
    
    def load_data(self, filename):
        """Load data."""
        with open(filename, 'r') as file:
            for line in file:
                hypernym, hyponym, count = line.split()
                self.hypernym_vocab.append(hypernym)
                self.hyponym_vocab.append(hyponym)
                self.data[(hypernym, hyponym)] = int(count)
        
        # Remove duplicate words
        self.hypernym_vocab = set(self.hypernym_vocab)
        self.hyponym_vocab = set(self.hyponym_vocab)

        self.hypernym_vocab = tuple(self.hypernym_vocab)
        self.hyponym_vocab = tuple(self.hyponym_vocab)

        self.hypernym_vocabulary_size = len(self.hypernym_vocab)
        self.hyponym_vocabulary_size = len(self.hyponym_vocab)

        # Dictionary which maps word to its index.
        self.word_hypernym_ix = {word: i for i, word in enumerate(self.hypernym_vocab)}
        self.word_hyponym_ix = {word: i for i, word in enumerate(self.hyponym_vocab)}

    class Net(nn.Module):
        """Network for training term embeddings."""
        def __init__(self, hypernym_vocabulary_size, hyponym_vocabulary_size, embedding_size):
            super(EmbeddingTrainer.Net, self).__init__()

            self.hypernym_vocabulary_size = hypernym_vocabulary_size
            self.hyponym_vocabulary_size = hyponym_vocabulary_size
            self.embedding_size = embedding_size

            self.hypernym_embedding = nn.Embedding(self.hypernym_vocabulary_size, self.embedding_size)
            self.hyponym_embedding = nn.Embedding(self.hyponym_vocabulary_size, self.embedding_size)

        def forward(self, hypernym_ix, hyponym_ix, count):
            embed_hypernym = self.hypernym_embedding(hypernym_ix)
            embed_hyponym = self.hyponym_embedding(hyponym_ix)
            norm = (embed_hypernym - embed_hyponym).norm(p=1)
            count = Variable(torch.Tensor([count]))
            output = norm + count.log1p()
            return output

    def train(self, epochs=10, print_loss=True):
        """Train the network"""

        net = EmbeddingTrainer.Net(self.hypernym_vocabulary_size, self.hyponym_vocabulary_size, self.embedding_size)

        optimizer = optim.Adam(net.parameters(), lr=0.01)

        for epoch in range(epochs):
            cost = Variable(torch.Tensor([0]))

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
                    output = net(Variable(torch.LongTensor([self.word_hypernym_ix[hypernym]])), Variable(torch.LongTensor([self.word_hyponym_ix[hyponym]])), count)
                    output1 = net(Variable(torch.LongTensor([self.word_hypernym_ix[hypernym1]])), Variable(torch.LongTensor([self.word_hyponym_ix[hyponym1]])), count1)
                    cost += torch.clamp(output - output1, min=0)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            if print_loss:
                print("Epoch {}: {}".format(epoch + 1, cost.data[0]))

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

    def array_to_string(self, array):
        """Convert numpy array to string"""
        string = ' '.join(map(lambda x: str(x), array))
        return string


if __name__ == "__main__":
    trainer = EmbeddingTrainer(embedding_size=10)
    trainer.load_data(os.path.join('data', 'sample_data'))
    trainer.train(epochs=1000)
    trainer.save_embeddings(os.path.join('data', 'hypernym_embedding'),\
                             os.path.join('data', 'hyponym_embedding'))
    







