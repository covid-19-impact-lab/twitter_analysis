import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from transformers import BertModel
from transformers import BertTokenizer

DATA_PATH = "/home/janos/twitter-analysis/data/SpinningBytes/"

SEED = 1234

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

torch.set_num_threads(28)


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()["hidden_size"]

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[: max_input_length - 2]
    return tokens


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    # num_iterations = len(iterator)
    print(f"Epoch: {iterator.epoch}")
    E = "="
    D = "."

    for batch in iterator:
        a = iterator.iterations
        s = len(iterator)
        print(f"{a}/{s}: [{a * E}>{(s-a) * D}]", end="\r")

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_training_data(data_path):
    df = pd.read_csv(data_path, sep="\t")
    print(df)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer.tokenize("hi, how are you? I am fine!")
indexes = tokenizer.convert_tokens_to_ids(tokens)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]

TEXT = data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_cut,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_idx,
    eos_token=eos_token_idx,
    pad_token=pad_token_idx,
    unk_token=unk_token_idx,
)

LABEL = data.LabelField(dtype=torch.float)

TWEET = data.Field()
SENTIMENT = data.Field()
fields = {"text": ("text", TEXT), "sentiment": ("label", LABEL)}
train_data, test_data = data.TabularDataset.splits(
    path=DATA_PATH, train="train.json", test="test.json", format="json", fields=fields
)

# print(vars(train_data[0]))
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")
# print(vars(train_data.examples[6]))

LABEL.build_vocab(train_data)  # TODO: order important?
print(LABEL.vocab.stoi)

BATCH_SIZE = 128

device = torch.device("cpu")

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    device=device,
)


bert = BertModel.from_pretrained("bert-base-uncased")

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

print(f"The model has {count_parameters(model):,} trainable parameters")


optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 5

best_valid_loss = float("inf")

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(model.state_dict(), "tut6-model.pt")

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
