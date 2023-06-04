import pickle
'''
with open("song_data_ix/dense_train_ix_user_song.dict", 'rb') as handle:
    train_user_song = pickle.load(handle)
with open("song_data_ix/dense_test_ix_user_song.dict", 'rb') as handle:
    test_user_song = pickle.load(handle)
with open("song_data_ix/dense_ix_song_user.dict", 'rb') as handle:
    full_song_user = pickle.load(handle)

pickle.dump(train_user_song, open("song_data_ix/dense_train_ix_user_song_py2.pkl","wb"), protocol=2)
pickle.dump(test_user_song, open("song_data_ix/dense_test_ix_user_song_py2.pkl","wb"), protocol=2)
pickle.dump(full_song_user, open("song_data_ix/dense_ix_song_user_py2.pkl","wb"), protocol=2)

with open("song_data_ix/rs_train_ix_user_song.dict", 'rb') as handle:
    train_user_song = pickle.load(handle)
with open("song_data_ix/rs_test_ix_user_song.dict", 'rb') as handle:
    test_user_song = pickle.load(handle)
with open("song_data_ix/rs_ix_song_user.dict", 'rb') as handle:
    full_song_user = pickle.load(handle)

pickle.dump(train_user_song, open("song_data_ix/rs_train_ix_user_song_py2.pkl","wb"), protocol=2)
pickle.dump(test_user_song, open("song_data_ix/rs_test_ix_user_song_py2.pkl","wb"), protocol=2)
pickle.dump(full_song_user, open("song_data_ix/rs_ix_song_user_py2.pkl","wb"), protocol=2)
'''
with open("song_data_ix/dense_ix_song_person.dict", 'rb') as handle:
    full_song_person = pickle.load(handle)
pickle.dump(full_song_person, open("song_data_ix/dense_ix_song_person_py2.pkl","wb"), protocol=2)

with open("song_data_ix/rs_ix_song_person.dict", 'rb') as handle:
    full_song_person = pickle.load(handle)
pickle.dump(full_song_person, open("song_data_ix/rs_ix_song_person_py2.pkl","wb"), protocol=2)
