import pickle
import numpy as np
from random import randint

def convert_for_bpr(pos_list, neg_list):
    '''
    converts pos/neg usersong pair lists into a matrix where every row contains
    101 tuples with the format ((user, song), 1 or 0)
    each row has 1 positive interactions and 100 negative interactions
    '''
    bpr_matrix = []
    total_row = len(pos_list)
    one_percent = total_row//100
    percent = 0
    pos_count = 0
    neg_count = 0
    # Don't pop. Just iterate.
    for tuple in pos_list:
        row = []
        for i in range(100):
            neg_interaction = neg_list[neg_count]
            row.append((neg_interaction, 0))
            neg_count += 1
        pos_interaction = pos_list[pos_count]
        row.insert(randint(0, 99), (pos_interaction, 1))
        bpr_matrix.append(row)
        pos_count += 1
        if pos_count % (total_row//100) == 0:
            percent += 1
            print(percent, ' percent done')

    # pickle to python2 format
    pickle.dump(bpr_matrix, open("../data/song_data/bpr_matrix_test_dense_py2.pkl","wb"), protocol=2)


def main():
    with open("../data/song_data_ix/dense_test_pos_interactions.txt", 'rb') as handle:
        test_pos_user_song = pickle.load(handle)
    with open("../data/song_data_ix/dense_test_neg_interactions.txt", 'rb') as handle:
        test_neg_user_song = pickle.load(handle)
    
    convert_for_bpr(test_pos_user_song, test_neg_user_song)

if __name__ == "__main__":
    main()
