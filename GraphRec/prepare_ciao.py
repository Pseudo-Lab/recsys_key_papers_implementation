from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

ciao_rate_file = loadmat("./data/ciao/rating.mat")
ciao_network_file = loadmat("./data/ciao/trustnetwork.mat")
ciao_rate_array = np.array(ciao_rate_file['rating'])
ciao_network_array = np.array(ciao_network_file['trustnetwork'])

ciao_rate_train, ciao_rate_test = train_test_split(ciao_rate_array, test_size = 0.3)

'''
data_file = open("./GraphRec/data/toy_dataset.pickle", 'rb')
history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)
print(social_adj_lists)
'''

history_u_lists, history_ur_lists = {}, {}
history_v_lists, history_vr_lists = {}, {}
len_train, len_test = len(ciao_rate_train), len(ciao_rate_test)
train_u, train_v, train_r = [0]*len_train, [0]*len_train, [0.]*len_train
test_u, test_v, test_r = [0]*len_test, [0]*len_test, [0.]*len_test
social_adj_lists = {}
ratings_list = {0.0: 0., 1.0: 1., 2.0: 2., 3.0: 3., 4.0: 4., 5.0: 5.}

for i in range(len_train):
    user_id = ciao_rate_train[i,0]
    item_id = ciao_rate_train[i,1]
    rating = ciao_rate_train[i,3]
    history_u_lists[user_id] = history_u_lists.get(user_id,[]) + [item_id]
    history_ur_lists[user_id] = history_ur_lists.get(user_id,[]) + [rating]
    history_v_lists[item_id] = history_v_lists.get(item_id,[]) + [user_id]
    history_vr_lists[item_id] = history_vr_lists.get(item_id,[]) + [rating]
    train_u[i], train_v[i], train_r[i] = user_id, item_id, rating

for i in range(len_test):
    user_id = ciao_rate_test[i,0]
    item_id = ciao_rate_test[i,1]
    rating = ciao_rate_test[i,3]
    test_u[i], test_v[i], test_r[i] = user_id, item_id, rating

for i in range(len(ciao_network_array)):
    user_1 = ciao_network_array[i,0]
    user_2 = ciao_network_array[i,1]
    social_adj_lists[user_1] = social_adj_lists.get(user_1, set())|{user_2}
    social_adj_lists[user_2] = social_adj_lists.get(user_2, set())|{user_1}

print(social_adj_lists)

print(history_u_lists)

with open("./data/ciao.pickle", "wb") as f:
    datas = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list)
    pickle_dump = pickle.dump(datas, f)