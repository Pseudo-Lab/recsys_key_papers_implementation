import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np

from torch.utils.data import Dataset

class BertTrainDataset(Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

class BertEvalDataset(Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        # self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        # negs = self.negative_samples[user]

        candidates = answer # + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

if __name__ == "__main__":
    import random

    df = pd.read_csv("../data/ml-1m/ml-1m.txt", sep=" ")
    df.columns = ["userId", "movieId"]
    
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    movieIdMapping = {k:i+2 for i, k in enumerate(df.movieId.unique())}
    inverseMovieIdMapping = {v:k for k, v in movieIdMapping.items()}

    df.movieId = df.movieId.map(movieIdMapping)


    user_group = df.groupby(by="userId").agg(list)
    
    train, val, test = {}, {}, {}

    for user in user_ids:
        items = user_group['movieId'][user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]       

    max_len = 100
    mask_prob = 0.1
    rng = random.Random(777)
    item_count = len(movieIdMapping)
    mask_token = item_count + 1

    dataset = BertTrainDataset(train, max_len, mask_prob, mask_token, item_count, rng)

    print(dataset.__getitem__(1))