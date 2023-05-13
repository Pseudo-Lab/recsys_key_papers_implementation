# %%
from tqdm import tqdm 
import os 

import numpy as np 
import pandas as pd

import scipy.sparse  as sp 

from sklearn.model_selection import train_test_split 

import torch 
from torch import nn, optim 
from torch.utils.data import Dataset, DataLoader 

# %%
class args:
    seed = 42
    num_layers = 3
    batch_size= 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = 'Parameters'

# %%
d_set = pd.read_csv('dataset/sample.csv', encoding='utf-8-sig')

# %%
d_train, d_test = train_test_split(d_set, train_size=0.6, random_state=args.seed)
d_valid, d_test = train_test_split(d_test, train_size=0.5, random_state=args.seed)

# %%
d_train = d_train.astype({'user_id':'category', 'business_id':'category'})
d_valid = d_valid.astype({'user_id':'category', 'business_id':'category'})
d_test = d_test.astype({'user_id':'category', 'business_id':'category'})

# %%
u_cat = d_train.user_id.cat.categories
b_cat = d_train.business_id.cat.categories

# %%
d_valid.user_id = d_valid.user_id.cat.set_categories(u_cat)
d_valid.business_id = d_valid.business_id.cat.set_categories(b_cat)

d_test.user_id = d_test.user_id.cat.set_categories(u_cat)
d_test.business_id = d_test.business_id.cat.set_categories(b_cat)

# %%
d_train.user_id = d_train.user_id.cat.codes
d_train.business_id = d_train.business_id.cat.codes 

d_valid.user_id = d_valid.user_id.cat.codes
d_valid.business_id = d_valid.business_id.cat.codes 

d_test.user_id = d_test.user_id.cat.codes
d_test.business_id = d_test.business_id.cat.codes 

# %%
d_train = d_train.dropna()
d_valid = d_valid.dropna()
d_test = d_test.dropna()

d_train.reset_index(drop=True, inplace=True)
d_valid.reset_index(drop=True, inplace=True)
d_test.reset_index(drop=True, inplace=True)

# %%
d_train = d_train.astype({'user_id': int, 'business_id': int})
d_valid = d_valid.astype({'user_id': int, 'business_id': int})
d_test = d_test.astype({'user_id': int, 'business_id': int})

# %%
args.num_users = d_train.user_id.max() + 1
args.num_items = d_train.business_id.max() + 1
args.latent_dim = 64
args.num_epochs = 50

# %%
class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats 

        self.W1 = nn.Linear(in_feats, out_feats)
        self.W2 = nn.Linear(in_feats, out_feats)

    def forward(self, L, SelfLoop, feats):
        # (L+I)EW_1
        sf_L = L + SelfLoop
        L = L.cuda()
        sf_L = sf_L.cuda()
        sf_E = torch.sparse.mm(sf_L, feats)
        left_part = self.W1(sf_E) # left part

        # EL odot EW_2, odot indicates element-wise product 
        LE = torch.sparse.mm(L, feats)
        E = torch.mul(LE, feats)
        right_part = self.W2(E)

        return left_part + right_part 

class NGCF(nn.Module):
    def __init__(self, args, matrix):
        super(NGCF, self).__init__()
        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.latent_dim = args.latent_dim 
        self.device = args.device

        self.user_emb = nn.Embedding(self.num_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim)

        self.num_layers = args.num_layers
        self.L = self.LaplacianMatrix(matrix)
        self.I = self.SelfLoop(self.num_users + self.num_items)

        self.leakyrelu = nn.LeakyReLU()
        self.GNNLayers = nn.ModuleList()

        for i in range(self.num_layers-1):
            self.GNNLayers.append(GNNLayer(self.latent_dim, self.latent_dim))

        self.fc_layer = nn.Sequential(
            nn.Linear(self.latent_dim * self.num_layers * 2, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )

    def SelfLoop(self, num):
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i, val)

    def LaplacianMatrix(self, ratings):
        iids = ratings['business_id'] + self.num_users 
        matrix = sp.coo_matrix((ratings['stars'], (ratings['user_id'], ratings['business_id'])))
        
        upper_matrix = sp.coo_matrix((ratings['stars'], (ratings['user_id'], iids)))
        lower_matrix = matrix.transpose()
        lower_matrix.resize((self.num_items, self.num_users + self.num_items))

        A = sp.vstack([upper_matrix, lower_matrix])
        row_sum = (A > 0).sum(axis=1)
        # row_sum = np.array(row_sum).flatten()
        diag = list(np.array(row_sum.flatten())[0])
        D = np.power(diag, -0.5)
        D = sp.diags(D)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row 
        col = L.col
        idx = np.stack([row, col])
        idx = torch.LongTensor(idx)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(idx, data)
        return SparseL 

    def FeatureMatrix(self):
        uids = torch.LongTensor([i for i in range(self.num_users)]).to(self.device)
        iids = torch.LongTensor([i for i in range(self.num_items)]).to(self.device)
        user_emb = self.user_emb(uids)
        item_emb = self.item_emb(iids)
        features = torch.cat([user_emb, item_emb], dim=0)
        return features

    def forward(self, uids, iids):
        iids = self.num_users + iids 

        features = self.FeatureMatrix()
        final_emb = features.clone()

        for gnn in self.GNNLayers:
            features = gnn(self.L, self.I, features)
            features = self.leakyrelu(features)
            final_emb = torch.concat([final_emb, features],dim=-1)

        user_emb = final_emb[uids]
        item_emb = final_emb[iids]

        inputs = torch.concat([user_emb, item_emb], dim=-1)
        outs = self.fc_layer(inputs)
        return outs.flatten()

# %%
class GraphDataset(Dataset):
    def __init__(self, dataframe):
        super(Dataset, self).__init__()
        
        self.uid = list(dataframe['user_id'])
        self.iid = list(dataframe['business_id'])
        self.ratings = list(dataframe['stars'])
    
    def __len__(self):
        return len(self.uid)
    
    def __getitem__(self, idx):
        uid = self.uid[idx]
        iid = self.iid[idx]
        rating = self.ratings[idx]
        
        return (uid, iid, rating)

# %%
def get_loader(args, dataset, num_workers):
    d_set = GraphDataset(dataset)
    return DataLoader(d_set, batch_size=args.batch_size, num_workers=num_workers)

# %%
train_loader = get_loader(args, d_train, 4)
valid_loader = get_loader(args, d_valid, 4)
test_loader = get_loader(args, d_test, 4)

# %%
def graph_evaluate(args, model, test_loader, criterion):
    output = []
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='evaluating...'):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'uids':   batch[0], 
                      'iids':   batch[1]}
            gold_y = batch[2].float()
            
            pred_y = model(**inputs)
            output.append(pred_y)
            
            loss = criterion(pred_y, gold_y)
            loss = torch.sqrt(loss)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss, output

# %%
def graph_train(args, model, train_loader, valid_loader, optimizer, criterion):
    best_loss = float('inf')
    train_losses, valid_losses = [], []
    for epoch in range(1, args.num_epochs + 1):
        train_loss = 0.0

        model.train()
        for batch in tqdm(train_loader, desc='training...'):
            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'uids':   batch[0], 
                      'iids':   batch[1]}
            
            gold_y = batch[2].float()
            

            pred_y = model(**inputs)
            
            loss = criterion(pred_y, gold_y)
            loss = torch.sqrt(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        valid_loss , outputs = graph_evaluate(args, model, valid_loader, criterion)
        valid_losses.append(valid_loss)
        

        print(f'Epoch: [{epoch}/{args.num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}')

        if best_loss > valid_loss:
            best_loss = valid_loss
            if not os.path.exists(args.SAVE_PATH):
                os.makedirs(args.SAVE_PATH)
            torch.save(model.state_dict(), os.path.join(args.SAVE_PATH, f'{model._get_name()}_parameters.pt'))

    return {
        'train_loss': train_losses, 
        'valid_loss': valid_losses
    }, outputs

# %%
models = NGCF(args, d_train).to(args.device)

optimizer = optim.Adam(models.parameters(), lr = 1e-3)
criterion = nn.L1Loss()

# %%
results = graph_train(args, models, train_loader, valid_loader, optimizer, criterion)

# %%
torch.cuda.get_arch_list()

# %%


# %%



