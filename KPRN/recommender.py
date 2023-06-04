import pickle
import torch
import argparse
import random
import mmap
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from os import mkdir
import pandas as pd
import numpy as np

import constants.consts as consts
from model import KPRN, train, predict
from data.format import format_paths
from data.path_extraction import find_paths_user_to_songs
from eval import hit_at_k, ndcg_at_k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='whether to train the model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--find_paths',
                        default=False,
                        action='store_true',
                        help='whether to find paths (otherwise load from disk)')
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to load data from')
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from')
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training ')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths')
    parser.add_argument('--user_limit',
                        type=int,
                        default=10,
                        help='max number of users to find paths for')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=5,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--lr',
                        type=float,
                        default=.002,
                        help='learning rate')
    parser.add_argument('--l2_reg',
                        type=float,
                        default=.0001,
                        help='l2 regularization coefficent')
    parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help='gamma for weighted pooling')
    parser.add_argument('--no_rel',
                        default=False,
                        action='store_true',
                        help='Run the model without relation if True')
    parser.add_argument('--np_baseline',
                        default=False,
                        action='store_true',
                        help='Run the model with the number of path baseline if True')
    parser.add_argument('--samples',
                        type=int,
                        default=-1,
                        help='number of paths to sample for each interaction (-1 means include all paths)')

    return parser.parse_args()


def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_data(song_person, person_song, user_song_all, song_user_all,
              song_user_split, user_song_split, neg_samples, e_to_ix, t_to_ix,
              r_to_ix, kg_path_file, len_3_branch, len_5_branch, limit=10, version="train", samples=-1):
    '''
    Constructs paths for train/test data,

    For training, we write each formatted interaction to a file as we find them
    For testing, for each combo of a pos paths and 100 neg paths we store these in a single line in the file
    '''
    path_dir = 'data/' + consts.PATH_DATA_DIR
    create_directory(path_dir)
    path_file = open(path_dir + kg_path_file, 'w')

    #trackers for statistics
    pos_paths_not_found = 0
    total_pos_interactions = 0
    num_neg_interactions = 0
    avg_num_pos_paths, avg_num_neg_paths = 0, 0

    for user,pos_songs in tqdm(list(user_song_split.items())[:limit]):
        total_pos_interactions += len(pos_songs)
        song_to_paths, neg_songs_with_paths = None, None
        cur_index = 0 #current index in negative list for adding negative interactions

        for pos_song in pos_songs:
            interactions = [] #just used with finding test paths
            if song_to_paths is None:
                if version == "train":
                    song_to_paths = find_paths_user_to_songs(user, song_person, person_song,
                                                                  song_user_split, user_song_split, 3, len_3_branch)
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song,
                                                                 song_user_split, user_song_split, 5, len_5_branch)
                else: #for testing we use entire song_user and user_song dictionaries
                    song_to_paths = find_paths_user_to_songs(user, song_person, person_song,
                                                                  song_user_all, user_song_all, 3, len_3_branch)
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song,
                                                                 song_user_all, user_song_all, 5, len_5_branch)
                for song in song_to_paths_len5.keys():
                    song_to_paths[song].extend(song_to_paths_len5[song])

                #select negative paths
                all_pos_songs = set(user_song_all[user])
                songs_with_paths = set(song_to_paths.keys())
                neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))

                top_neg_songs = neg_songs_with_paths
                random.shuffle(top_neg_songs)

            #add paths for positive interaction
            pos_paths = song_to_paths[pos_song]
            if len(pos_paths) > 0:
                if samples != -1:
                    pos_paths = sample_paths(pos_paths, samples)
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1)
                if version == "train":
                    path_file.write(repr(interaction) + "\n")
                else:
                    interactions.append(interaction)
                avg_num_pos_paths += len(pos_paths)
            else:
                pos_paths_not_found += 1
                continue #don't add interactions with no positive paths

            #add negative interactions that have paths
            found_all_samples = True
            for i in range(neg_samples):
                #check if not enough neg paths
                if cur_index >= len(top_neg_songs):
                    print("not enough neg paths, only found:", str(i))
                    found_all_samples = False
                    break
                neg_song = top_neg_songs[cur_index]
                neg_paths = song_to_paths[neg_song]

                if samples != -1:
                    neg_paths = sample_paths(neg_paths, samples)
                interaction = (format_paths(neg_paths, e_to_ix, t_to_ix, r_to_ix), 0)
                if version == "train":
                    path_file.write(repr(interaction) + "\n")
                else:
                    interactions.append(interaction)

                avg_num_neg_paths += len(neg_paths)
                num_neg_interactions += 1
                cur_index += 1

            if found_all_samples and version == "test":
                path_file.write(repr(interactions) + "\n")

    avg_num_neg_paths = avg_num_neg_paths / num_neg_interactions
    avg_num_pos_paths = avg_num_pos_paths / (total_pos_interactions - pos_paths_not_found)

    print("number of pos paths attempted to find:", total_pos_interactions)
    print("number of pos paths not found:", pos_paths_not_found)
    print("avg num paths per positive interaction:", avg_num_pos_paths)
    print("avg num paths per negative interaction:", avg_num_neg_paths)

    path_file.close()
    return


def load_string_to_ix_dicts(network_type):
    '''
    Loads the dictionaries mapping entity, relation, and type to id
    '''
    data_path = 'data/' + consts.SONG_IX_MAPPING_DIR + network_type

    with open(data_path + '_type_to_ix.dict', 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(data_path + '_relation_to_ix.dict', 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(data_path + '_entity_to_ix.dict', 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_rel_ix_dicts(network_type):
    '''
    Loads the relation dictionaries
    '''
    data_path = 'data/' + consts.SONG_IX_DATA_DIR + network_type

    with open(data_path + '_ix_song_person.dict', 'rb') as handle:
        song_person = pickle.load(handle)
    with open(data_path + '_ix_person_song.dict', 'rb') as handle:
        person_song = pickle.load(handle)
    with open(data_path + '_ix_song_user.dict', 'rb') as handle:
        song_user = pickle.load(handle)
    with open(data_path + '_ix_user_song.dict', 'rb') as handle:
        user_song = pickle.load(handle)

    return song_person, person_song, song_user, user_song


def main():
    '''
    Main function for kprn model testing and training
    '''
    print("Main Loaded")
    random.seed(1)
    args = parse_args()
    model_path = "model/" + args.model

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts(args.subnetwork)
    song_person, person_song, song_user, user_song = load_rel_ix_dicts(args.subnetwork)

    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TAG_SIZE, args.no_rel)

    data_ix_path = 'data/' + consts.SONG_IX_DATA_DIR + args.subnetwork

    if args.train:
        print("Training Starting")
        #either load interactions from disk, or run path extraction algorithm
        if args.find_paths:
            print("Finding paths")

            with open(data_ix_path + '_train_ix_user_song.dict', 'rb') as handle:
                user_song_train = pickle.load(handle)
            with open(data_ix_path + '_train_ix_song_user.dict', 'rb') as handle:
                song_user_train = pickle.load(handle)

            load_data(song_person, person_song, user_song, song_user,
                      song_user_train, user_song_train, consts.NEG_SAMPLES_TRAIN,
                      e_to_ix, t_to_ix, r_to_ix, args.kg_path_file, consts.LEN_3_BRANCH,
                      consts.LEN_5_BRANCH_TRAIN, limit=args.user_limit, version="train", samples=args.samples)

        model = train(model, args.kg_path_file, args.batch_size, args.epochs, model_path,
                      args.load_checkpoint, args.not_in_memory, args.lr, args.l2_reg, args.gamma, args.no_rel)

    if args.eval:
        print("Evaluation Starting")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is", device)

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        model = model.to(device)

        if args.find_paths:
            print("Finding Paths")

            with open(data_ix_path + '_test_ix_user_song.dict', 'rb') as handle:
                user_song_test = pickle.load(handle)
            with open(data_ix_path + '_test_ix_song_user.dict', 'rb') as handle:
                song_user_test = pickle.load(handle)

            load_data(song_person, person_song, user_song, song_user,
                      song_user_test, user_song_test, consts.NEG_SAMPLES_TEST,
                      e_to_ix, t_to_ix, r_to_ix, args.kg_path_file, consts.LEN_3_BRANCH,
                      consts.LEN_5_BRANCH_TEST, limit=args.user_limit, version="test", samples=args.samples)

        #predict scores using model for each combination of one pos and 100 neg interactions
        hit_at_k_scores = defaultdict(list)
        ndcg_at_k_scores = defaultdict(list)
        if args.np_baseline:
            num_paths_baseline_hit_at_k = defaultdict(list)
            num_paths_baseline_ndcg_at_k = defaultdict(list)
        max_k = 15

        file_path = 'data/path_data/' + args.kg_path_file
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))
                prediction_scores = predict(model, test_interactions, args.batch_size, device, args.no_rel, args.gamma)
                target_scores = [x[1] for x in test_interactions]

                #merge prediction scores and target scores into tuples, and rank
                merged = list(zip(prediction_scores, target_scores))
                s_merged = sorted(merged, key=lambda x: x[0], reverse=True)

                for k in range(1,max_k+1):
                    hit_at_k_scores[k].append(hit_at_k(s_merged, k))
                    ndcg_at_k_scores[k].append(ndcg_at_k(s_merged, k))

                #Baseline of ranking based on number of paths
                if args.np_baseline:
                    random.shuffle(test_interactions)
                    s_inters = sorted(test_interactions, key=lambda x: len(x[0]), reverse=True)
                    for k in range(1,max_k+1):
                        num_paths_baseline_hit_at_k[k].append(hit_at_k(s_inters, k))
                        num_paths_baseline_ndcg_at_k[k].append(ndcg_at_k(s_inters, k))

        scores = []

        for k in hit_at_k_scores.keys():
            hit_at_ks = hit_at_k_scores[k]
            ndcg_at_ks = ndcg_at_k_scores[k]
            print()
            print(["Average hit@K for k={0} is {1:.4f}".format(k, mean(hit_at_ks))])
            print(["Average ndcg@K for k={0} is {1:.4f}".format(k, mean(ndcg_at_ks))])
            scores.append([args.model, args.kg_path_file, k, mean(hit_at_ks), mean(ndcg_at_ks)])

        if args.np_baseline:
            for k in hit_at_k_scores.keys():
                print()
                print(["Num Paths Baseline hit@K for k={0} is {1:.4f}".format(k, mean(num_paths_baseline_hit_at_k[k]))])
                print(["Num Paths Baseline ndcg@K for k={0} is {1:.4f}".format(k, mean(num_paths_baseline_ndcg_at_k[k]))])
                scores.append(['np_baseline', args.kg_path_file, k, mean(num_paths_baseline_hit_at_k[k]), mean(num_paths_baseline_ndcg_at_k[k])])

        # saving scores
        scores_cols = ['model', 'test_file', 'k', 'hit', 'ndcg']
        scores_df = pd.DataFrame(scores, columns = scores_cols)
        scores_path = 'model_scores.csv'
        try:
            model_scores = pd.read_csv(scores_path)
        except FileNotFoundError:
            model_scores = pd.DataFrame(columns=scores_cols)
        model_scores=model_scores.append(scores_df, ignore_index = True, sort=False)
        model_scores.to_csv(scores_path,index=False)


if __name__ == "__main__":
    main()
