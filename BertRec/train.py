import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader


from bert_modules.bert import BERT4Rec
from data_loader import BertTrainDataset, BertEvalDataset
from metrics_utils import recalls_and_ndcgs_for_ks

def main():
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

    heads = 4
    layers = 6
    emb_dim = 512
    dropout = 0.1
    epochs = 20
    batch_size = 256

    train_dataset = BertTrainDataset(train, max_len, mask_prob, mask_token, item_count, rng)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = BertEvalDataset(train, val, max_len, mask_token)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = BERT4Rec(max_len, item_count, layers, heads, emb_dim, dropout)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss(ignore_index=0)

    metric_ks = [ 1, 5, 10, 20, 50, 100]

    for epoch in range(1, epochs + 1):

        for batch in tqdm(train_dataloader):
            batch_size = batch[0].size(0)
            batch = [ x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            seqs, labels = batch
            logits = model(seqs)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T

            loss = ce(logits, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                    batch = [ x.to(device) for x in batch]
                    seqs, candidates, labels = batch
                    logits = model(seqs)

                    scores = scores[:, -1, :]
                    scores = scores.gather(1, candidates)

                    metrics = recalls_and_ndcgs_for_ks(scores, labels, metric_ks)
                    print(metrics)
            break
        break

    # for epoch in range(1, epochs + 1):
    #     total_loss = 0
    #     train_acc = []
    #     train_bs = [] 

    #     for batch in tqdm(train_dataloader):
    #         source = batch["source"].to(device)
    #         source_mask = batch["source_mask"].to(device)
    #         target = batch["target"].to(device)
    #         target_mask = batch["target_mask"].to(device)
    #         train_bs += [source.size(0)]

    #         mask = source == 1

    #         optimizer.zero_grad()

    #         output = model(source, source_mask)

    #         loss = F.cross_entropy(output, target)
    #         loss = loss * mask
    #         loss = loss.sum() / (mask.sum() + 1e-8)

    #         total_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()

    #         _, pred = output.max(1)
        
    #         y_true = torch.masked_select(target, mask)
    #         pred = torch.masked_select(pred, mask)

            
    #         mean = (y_true == pred).double().mean()
    #         train_acc += [mean.item()]

    #     epoch_acc = (torch.sum(torch.tensor(train_bs) * torch.tensor(train_acc)) / torch.sum(torch.tensor(train_bs))) * 100

    #     print(f"train_loss : {total_loss / len(train_dataloader)}\ntrain_acc : {epoch_acc.item()}")

if __name__ == "__main__":
    main()

