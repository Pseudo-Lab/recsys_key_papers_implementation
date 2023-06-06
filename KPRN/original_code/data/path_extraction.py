import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath('./constants'))))

import pickle
import random
import constants.consts as consts
from collections import defaultdict
import copy


class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # array of [entity, entity type, relation to next] triplets
        self.length = length
        self.entities = entities    # set to keep track of the entities alr in the path to avoid cycles

def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


def find_paths_user_to_songs(start_user, song_person, person_song, song_user, user_song, max_length, sample_nums):
    '''
    Finds sampled paths of max depth from a user to a sampling of songs
    '''
    song_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        #add path to song_to_user_paths dict, just want paths of max_length rn since either length 3 or 5
        if type == consts.SONG_TYPE and front.length == max_length:
            song_to_paths[entity].append(front.path)

        if front.length == max_length:
            continue

        if type == consts.USER_TYPE and entity in user_song:
            song_list = user_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.USER_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)

        elif type == consts.SONG_TYPE:
            if entity in song_user:
                user_list = song_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        stack.append(new_state)
            if entity in song_person:
                person_list = song_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_PERSON_REL
                        new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{person})
                        stack.append(new_state)

        elif type == consts.PERSON_TYPE and entity in person_song:
            song_list = person_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.PERSON_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)

    return song_to_paths


def main():
    with open("song_data_ix/dense_ix_song_person.dict", 'rb') as handle:
        song_person = pickle.load(handle)

    with open("song_data_ix/dense_ix_person_song.dict", 'rb') as handle:
        person_song = pickle.load(handle)

    with open("song_data_ix/dense_ix_song_user.dict", 'rb') as handle:
        song_user = pickle.load(handle)

    with open("song_data_ix/dense_ix_user_song.dict", 'rb') as handle:
        user_song = pickle.load(handle)

    print(find_paths_user_to_songs(224218, song_person, person_song, song_user, user_song, 3, 1))


if __name__ == "__main__":
    main()
