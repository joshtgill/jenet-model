from torch import nn


class FNN(nn.Module):

    def __init__(self, vocab_size, encoding_size, output_size):
        super().__init__()

        embedding_size = int(vocab_size ** (1/2))
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(encoding_size * embedding_size, output_size),
        )


    def forward(self, x):
        return self.stack(self.flatten(self.embedding(x)))
