import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.utils.clip_grad as clip_grad_norm


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # one dimensional tensor that contains index of all words
        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, "r") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    # vector on indices that indicates the whole document is being created here
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1

        # find how many batches we need, double divison will ignore the remainder and remove the decimal part
        num_batches = rep_tensor.shape[0] // batch_size

        # remove the remainder
        rep_tensor = rep_tensor[: num_batches * batch_size]
        # return the reshaped tensor
        return rep_tensor.view(batch_size, -1)


# define the parameters
embedding_size = 128  # how many features that we need for LSTM, that is embedding size
hidden_size = 1024  # num of hidden units of LSTM
num_layers = 1  # single layer LSTM
num_epochs = 20
batch_size = 20
timesteps = 30  # look at 30 previous words to predict a word
learning_rate = 0.002


corpus = TextProcess()

rep_tensor = corpus.get_data("alice.txt", batch_size)
print(rep_tensor.shape)
vocab_size = len(corpus.dictionary)
print(vocab_size)

# at a time you need timesteps of word embeddings. So from each row or each data you are just picking a portion of the data in each timestep
num_batches = rep_tensor.shape[1] // timesteps
print(num_batches)


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x, h)
        # batch_size*timesteps, hidden_size
        out = self.fc(out.reshape(out.shape[0] * out.shape[1], out.shape[2]))
        return out, (h, c)


model = TextLSTM(vocab_size, embedding_size, hidden_size, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # hidden state and memory state here
    states = (
        torch.zeros(num_layers, batch_size, hidden_size),
        torch.zeros(num_layers, batch_size, hidden_size),
    )
    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        inputs = rep_tensor[:, i : i + timesteps]  # -> (:,0:0+30)... output -->(:,1:31)
        targets = rep_tensor[:, (i + 1) : (i + 1) + timesteps]
        outputs, _ = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        # clip gradient to avoid exploding gradient problem
        clip_grad_norm.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // timesteps
        if step % 100 == 0:
            print(f"Epoch: {epoch} Loss: {loss.item()}")


# Test the model
with torch.no_grad():
    with open("results.txt", "w") as f:
        # initialize the hidden state and memory state
        state = (
            torch.zeros(num_layers, 1, hidden_size),
            torch.zeros(num_layers, 1, hidden_size),
        )
        input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            output, _ = model(input, state)
            print(output.shape)
            # to get the probability-
            prob = output.exp()
            # get the word id
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            # replace input with sample word id for next time step
            input.fill_(word_id)

            # write the result file
            word = corpus.dictionary.idx2word[word_id]
            word = "\n" if word == "<eos>" else word + " "
            f.write(word)

            if (i + 1) % 100 == 0:
                print(
                    "Sampled [{}/500] words and save to {}".format(i + 1, "results.txt")
                )
