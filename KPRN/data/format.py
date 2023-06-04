import torch
import constants.consts as consts

'''
functions used for converting path data into format for KPRN model
'''

def format_paths(paths, e_to_ix, t_to_ix, r_to_ix):
    '''
    Pads paths up to max path length, converting each path into tuple
    of (padded_path, path length).
    '''

    new_paths = []
    for path in paths:
        path_len = len(path)
        pad_path(path, e_to_ix, t_to_ix, r_to_ix, consts.MAX_PATH_LEN, consts.PAD_TOKEN)
        new_paths.append((path, path_len))
    return new_paths

def find_max_train_length(data_tuples):
    '''
    Finds max path length in a list of (interaction, target) tuples
    '''
    max_len = 0
    for (paths, _) in data_tuples:
        for path in paths:
            max_len = max(len(path), max_len)
    return max_len

def pad_path(seq, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token):
    '''
    Pads paths up to max path length
    '''
    relation_padding =  r_to_ix[padding_token]
    type_padding = t_to_ix[padding_token]
    entity_padding = e_to_ix[padding_token]

    while len(seq) < max_len:
        seq.append([entity_padding, type_padding, relation_padding])

    return seq
