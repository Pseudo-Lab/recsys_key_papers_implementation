import pandas as pd
import pickle
import argparse
from collections import defaultdict
import random
from tqdm import tqdm

import sys
from os import path, mkdir

sys.path.append(path.dirname(path.dirname(path.abspath('./constants'))))
import constants.consts as consts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--songs_file',
                        default='songs.csv',
                        help='Path to the CSV file containing song information')
    parser.add_argument('--interactions_file',
                        default='train.csv',
                        help='Path to the CSV file containing user song interactions')
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to form from the full KG')
    return parser.parse_args()


def make_person_list(row):
    person_set = set()
    if not isinstance(row['artist_name'], float):
        for x in row['artist_name'].split('|'):
            person_set.add(x.strip())
    if not isinstance(row['composer'], float):
        for x in row['composer'].split('|'):
            person_set.add(x.strip())
    if not isinstance(row['lyricist'], float):
        for x in row['lyricist'].split('|'):
            person_set.add(x.strip())
    return list(person_set)


def song_data_prep(songs_file, interactions_file, export_dir):
    '''
    :return: Write out 4 python dictionaries for the edges of KG
    '''

    songs = pd.read_csv(songs_file)
    interactions = pd.read_csv(interactions_file)

    # song_person.dict
    # dict where key = song_id, value = list of persons (artists, composers, lyricists) of the song
    person = songs[['song_id', 'artist_name', 'composer', 'lyricist']]
    person_list = person.apply(lambda x: make_person_list(x), axis=1)
    song_person = pd.concat([songs['song_id'], person_list], axis=1)
    song_person.columns = ['song_id', 'person_list']
    song_person_dict = song_person.set_index('song_id')['person_list'].to_dict()
    with open(export_dir + consts.SONG_PERSON_DICT, 'wb') as handle:
        pickle.dump(song_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # person_song.dict
    # dict where key = a person, value = list of songs related to this person
    person_song_dict = {}
    for row in song_person.iterrows():
        for person in row[1]['person_list']:
            if person not in person_song_dict:
                person_song_dict[person] = []
            person_song_dict[person].append(row[1]['song_id'])
    with open(export_dir + consts.PERSON_SONG_DICT, 'wb') as handle:
        pickle.dump(person_song_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # song_user.dict
    # dict where key = song_id, value = list of user_ids
    song_user = interactions[['song_id', 'msno']].set_index('song_id').groupby('song_id')['msno'].apply(list).to_dict()
    # msno is the user_id
    with open(export_dir + consts.SONG_USER_DICT, 'wb') as handle:
        pickle.dump(song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_song.dict
    # dict where key = user_id, value = list of song_ids
    user_song = interactions[['msno', 'song_id']].set_index('msno').groupby('msno')['song_id'].apply(list).to_dict()
    with open(export_dir + consts.USER_SONG_DICT, 'wb') as handle:
        pickle.dump(user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_song_tuple.txt
    # numpy array of [user_id, song_id] pairs sorted in the order of user_id
    user_song_tuple = interactions[['msno', 'song_id']].sort_values(by='msno').to_string(header=False, index=False,
                                                                                         index_names=False).split('\n')
    user_song_tuple = [row.split() for row in user_song_tuple]
    with open('user_song_tuple.txt', 'wb') as handle:
        pickle.dump(user_song_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_subnetwork(network_type, dir, factor=0.1):
    if network_type == 'full':
        return

    # Load Data

    with open(dir + consts.SONG_USER_DICT, 'rb') as handle:
        song_user = pickle.load(handle)
    with open(dir + consts.USER_SONG_DICT, 'rb') as handle:
        user_song = pickle.load(handle)
    with open(dir + consts.SONG_PERSON_DICT, 'rb') as handle:
        song_person = pickle.load(handle)
    with open(dir + consts.PERSON_SONG_DICT, 'rb') as handle:
        person_song = pickle.load(handle)
    song_user = defaultdict(list, song_user)
    song_person = defaultdict(list, song_person)
    user_song = defaultdict(list, user_song)
    person_song = defaultdict(list, person_song)

    # Sort Nodes By Degree in descending order

    # key: song, value: number of users listening to it + number of person relating to its creation
    song_degree_dict = {}
    for (k, v) in song_user.items():
        song_degree_dict[k] = v
    for (k, v) in song_person.items():
        if k in song_degree_dict.keys():
            song_degree_dict[k] = song_degree_dict[k] + v
        else:
            song_degree_dict[k] = v
    song_degree = [(k, len(v)) for (k, v) in song_degree_dict.items()]
    song_degree.sort(key=lambda x: -x[1])

    # key: person, value: number of songs they create
    person_degree = [(k, len(v)) for (k, v) in person_song.items()]
    person_degree.sort(key=lambda x: -x[1])

    # key: user, value: number of songs they listen to
    user_degree = [(k, len(v)) for (k, v) in user_song.items()]
    user_degree.sort(key=lambda x: -x[1])

    # Construct Subnetworks

    # find the nodes
    print('finding the nodes...')
    if network_type == 'dense':
        song_nodes_holder = song_degree[:int(len(
            song_degree) * factor)]  # song_id is the first item in the tuple element of the returned list
        song_nodes = [node_holder[0] for node_holder in song_nodes_holder]

        user_nodes_holder = user_degree[:int(len(user_degree) * factor)]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[:int(len(person_degree) * factor)]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    elif network_type == 'rs':
        song_nodes_holder = random.sample(song_degree, int(len(
            song_degree) * factor))  # song_id is the first item in the tuple element of the returned list
        song_nodes = [node_holder[0] for node_holder in song_nodes_holder]

        user_nodes_holder = random.sample(user_degree, int(len(user_degree) * factor))
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = random.sample(person_degree, int(len(person_degree) * factor))
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    elif network_type == 'sparse':
        song_nodes_holder = song_degree[-int(len(
            song_degree) * factor):]  # song_id is the first item in the tuple element of the returned list
        song_nodes = [node_holder[0] for node_holder in song_nodes_holder]

        user_nodes_holder = user_degree[-int(len(user_degree) * factor):]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[-int(len(person_degree) * factor):]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    nodes = song_nodes + user_nodes + person_nodes
    print('The %s subnetwork has %d nodes: %d songs, %d users, %d persons.' % (network_type, \
                                                                               len(nodes), \
                                                                               len(song_nodes), \
                                                                               len(user_nodes), \
                                                                               len(person_nodes)))
    # find the edges
    # (node1, node2) and (node2, node1) both exist
    edges_type1 = []  # a list of pairs (song, user)
    edges_type2 = []  # a list of pairs (song, person)
    edges_type3 = []  # a list of pairs (user, song)
    edges_type4 = []  # a list of pairs (person, song)
    nodes_set = set(nodes)

    for i in tqdm(nodes_set):  # (node1, node2) and (node2, node1) both exist
        connect_1 = set(song_user[i]).intersection(nodes_set)
        for j in connect_1:
            edges_type1.append((i, j))

        connect_2 = set(song_person[i]).intersection(nodes_set)
        for j in connect_2:
            edges_type2.append((i, j))

        connect_3 = set(user_song[i]).intersection(nodes_set)
        for j in connect_3:
            edges_type3.append((i, j))

        connect_4 = set(person_song[i]).intersection(nodes_set)
        for j in connect_4:
            edges_type4.append((i, j))

    edges = edges_type1 + edges_type2 + edges_type3 + edges_type4
    print('The %s subnetwork has %d edges.' % (network_type, len(edges)))

    # Export the Subnetworks

    # <NETWORK_TYPE>_song_user.dict
    # key: song, value: a list of users
    song_user_dict = defaultdict(list)
    for edge in edges_type1:
        song = edge[0]
        user = edge[1]
        song_user_dict[song].append(user)
    song_user_dict = dict(song_user_dict)
    prefix = dir + network_type + '_'
    with open(prefix + consts.SONG_USER_DICT, 'wb') as handle:
        pickle.dump(song_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_song_person.dict
    # key: song, value: a list of persons
    song_person_dict = defaultdict(list)
    for edge in edges_type2:
        song = edge[0]
        person = edge[1]
        song_person_dict[song].append(person)
    song_person_dict = dict(song_person_dict)
    with open(prefix + consts.SONG_PERSON_DICT, 'wb') as handle:
        pickle.dump(song_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_user_song.dict
    # key: user, value: a list of songs
    user_song_dict = defaultdict(list)
    for edge in edges_type3:
        user = edge[0]
        song = edge[1]
        user_song_dict[user].append(song)
    user_song_dict = dict(user_song_dict)
    with open(prefix + consts.USER_SONG_DICT, 'wb') as handle:
        pickle.dump(user_song_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_person_song.dict
    # key: person, value: a list of songs
    person_song_dict = defaultdict(list)
    for edge in edges_type4:
        person = edge[0]
        song = edge[1]
        person_song_dict[person].append(song)
    person_song_dict = dict(person_song_dict)
    with open(prefix + consts.PERSON_SONG_DICT, 'wb') as handle:
        pickle.dump(person_song_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        key_id = entity_to_ix[(key, start_type)]
        value_ids = []
        for val in values:
            value_ids.append(entity_to_ix[(val, end_type)])
        new_rel[key_id] = value_ids
    return new_rel

def ix_mapping(network_type, import_dir, export_dir, mapping_export_dir):
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'person': consts.PERSON_TYPE, 'user': consts.USER_TYPE, 'song': consts.SONG_TYPE,
                  pad_token: consts.PAD_TYPE}
    relation_to_ix = {'song_person': consts.SONG_PERSON_REL, 'person_song': consts.PERSON_SONG_REL,
                      'user_song': consts.USER_SONG_REL, 'song_user': consts.SONG_USER_REL, '#UNK_RELATION': consts.UNK_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    # entity vocab set is combination of songs, users, and persons
    song_data_prefix = import_dir + network_type + '_'
    with open(song_data_prefix + consts.SONG_USER_DICT, 'rb') as handle:
        song_user = pickle.load(handle)
    with open(song_data_prefix + consts.SONG_PERSON_DICT, 'rb') as handle:
        song_person = pickle.load(handle)
    with open(song_data_prefix + consts.USER_SONG_DICT, 'rb') as handle:
        user_song = pickle.load(handle)
    with open(song_data_prefix + consts.PERSON_SONG_DICT, 'rb') as handle:
        person_song = pickle.load(handle)

    songs = set(song_user.keys()) | set(song_person.keys())
    users = set(user_song.keys())
    persons = set(person_song.keys())

    # Id-ix mappings
    entity_to_ix = {(song, consts.SONG_TYPE): ix for ix, song in enumerate(songs)}
    entity_to_ix.update({(user, consts.USER_TYPE): ix + len(songs) for ix, user in enumerate(users)})
    entity_to_ix.update(
        {(person, consts.PERSON_TYPE): ix + len(songs) + len(users) for ix, person in enumerate(persons)})
    entity_to_ix[pad_token] = len(entity_to_ix)

    # Ix-id mappings
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # Export mappings
    song_ix_mapping_prefix = mapping_export_dir + network_type + '_'
    # eg. song_ix_data/dense_type_to_ix.dict
    with open(song_ix_mapping_prefix + consts.TYPE_TO_IX, 'wb') as handle:
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(song_ix_mapping_prefix + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(song_ix_mapping_prefix + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(song_ix_mapping_prefix + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(song_ix_mapping_prefix + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(song_ix_mapping_prefix + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Update the KG
    song_user_ix = convert_to_ids(entity_to_ix, song_user, consts.SONG_TYPE, consts.USER_TYPE)
    user_song_ix = convert_to_ids(entity_to_ix, user_song, consts.USER_TYPE, consts.SONG_TYPE)
    song_person_ix = convert_to_ids(entity_to_ix, song_person, consts.SONG_TYPE, consts.PERSON_TYPE)
    person_song_ix = convert_to_ids(entity_to_ix, person_song, consts.PERSON_TYPE, consts.SONG_TYPE)

    # export
    # eg. song_ix_data/dense_ix_song_user.dict
    ix_prefix = export_dir + network_type + '_ix_'
    with open(ix_prefix + consts.SONG_USER_DICT, 'wb') as handle:
        pickle.dump(song_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.USER_SONG_DICT, 'wb') as handle:
        pickle.dump(user_song_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.SONG_PERSON_DICT, 'wb') as handle:
        pickle.dump(song_person_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.PERSON_SONG_DICT, 'wb') as handle:
        pickle.dump(person_song_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_test_split(network_type, dir):
    with open(dir + network_type + '_ix_' + consts.USER_SONG_DICT, 'rb') as handle:
        user_song = pickle.load(handle)

    # KG and positive
    train_user_song = {}
    test_user_song = {}
    train_song_user = defaultdict(list)
    test_song_user = defaultdict(list)

    for user in user_song:
        pos_songs = user_song[user]
        random.shuffle(pos_songs)
        cut = int(len(pos_songs) * 0.8)

        # train
        train_user_song[user] = pos_songs[:cut]
        for song in pos_songs[:cut]:
            train_song_user[song].append(user)

        # test
        test_user_song[user] = pos_songs[cut:]
        for song in pos_songs[cut:]:
            test_song_user[song].append(user)

    # Export
    # eg. song_ix_data/dense_train_ix_song_user.dict
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.USER_SONG_DICT), 'wb') as handle:
        pickle.dump(train_user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.USER_SONG_DICT), 'wb') as handle:
        pickle.dump(test_user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.SONG_USER_DICT), 'wb') as handle:
        pickle.dump(train_song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.SONG_USER_DICT), 'wb') as handle:
        pickle.dump(test_song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def main():
    print("Data preparation:")
    args = parse_args()

    network_prefix = args.subnetwork
    if network_prefix == 'full':
        network_prefix = ''

    print("Forming knowledge graph...")
    create_directory(consts.SONG_DATA_DIR)
    song_data_prep(consts.SONG_DATASET_DIR + args.songs_file,
                   consts.SONG_DATASET_DIR + args.interactions_file,
                   consts.SONG_DATA_DIR)

    print("Forming network...")
    find_subnetwork(args.subnetwork, consts.SONG_DATA_DIR)

    print("Mapping ids to indices...")
    create_directory(consts.SONG_IX_DATA_DIR)
    create_directory(consts.SONG_IX_MAPPING_DIR)
    ix_mapping(network_prefix, consts.SONG_DATA_DIR, consts.SONG_IX_DATA_DIR, consts.SONG_IX_MAPPING_DIR)

    print("Creating training and testing datasets...")
    train_test_split(network_prefix, consts.SONG_IX_DATA_DIR)

if __name__ == "__main__":
    main()
