import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import linecache

import constants.consts as consts

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import KPRN
from tqdm import tqdm
from statistics import mean


class TrainInteractionData(Dataset):
    '''
    Dataset that can either store all interaction data in memory or load it line
    by line when needed
    '''
    def __init__(self, train_path_file, in_memory=True):
        self.in_memory = in_memory
        self.file = 'data/path_data/' + train_path_file
        self.num_interactions = 0
        self.interactions = []
        if in_memory:
            with open(self.file, "r") as f:
                for line in f:
                    self.interactions.append(eval(line.rstrip("\n")))
            self.num_interactions = len(self.interactions)
        else:
            with open(self.file, "r") as f:
                for line in f:
                    self.num_interactions += 1

    def __getitem__(self, idx):
        #load the specific interaction either from memory or from file line
        if self.in_memory:
            return self.interactions[idx]
        else:
            line = linecache.getline(self.file, idx+1)
            return eval(line.rstrip("\n"))


    def __len__(self):
        return self.num_interactions


def my_collate(batch):
    '''
    Custom dataloader collate function since we have tuples of lists of paths
    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def sort_batch(batch, indexes, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    return seq_tensor, indexes_tensor, seq_lengths


def train(model, train_path_file, batch_size, epochs, model_path, load_checkpoint,
         not_in_memory, lr, l2_reg, gamma, no_rel):
    '''
    -trains and outputs a model using the input data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is", device)
    model = model.to(device)

    loss_function = nn.NLLLoss()

    # l2 regularization is tuned from {10−5 , 10−4 , 10−3 , 10−2 }, I think this is weight decay
    # Learning rate is found from {0.001, 0.002, 0.01, 0.02} with grid search
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    if load_checkpoint:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #DataLoader used for batches
    interaction_data = TrainInteractionData(train_path_file, in_memory=not not_in_memory)
    train_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print("Epoch is:", epoch+1)
        losses = []
        for interaction_batch, targets in tqdm(train_loader): #have tqdm here when not on colab
            #construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)


            #sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            #Pytorch accumulates gradients, so we need to clear before each instance
            model.zero_grad()

            #Run the forward pass.
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel)

            #Get weighted pooling of scores over interaction id groups
            start = True
            for i in range(len(interaction_batch)):
                #get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                #weighted pooled scores for this interaction
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                if start:
                    #unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            prediction_scores = F.log_softmax(pooled_scores, dim=1)

            #Compute the loss, gradients, and update the parameters by calling .step()
            loss = loss_function(prediction_scores.to(device), targets.to(device))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("loss is:", mean(losses))
        #Save model to disk
        print("Saving checkpoint to disk...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
          }, model_path)
        #torch.save(model.state_dict(), model_path)

    return model
