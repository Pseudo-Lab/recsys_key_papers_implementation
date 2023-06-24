'''
1. 10번 이상 평가된 item(영화)만 남김
2. Remap item ids
3. Remap user ids
4. Get user session
5. train, test split
'''


import pandas as pd


def make_datasets(data, len_seq, len_tag, len_pred):

    # 1. 10번 이상 평가된 item(영화)만 남김
    p = data.groupby('item')['user'].count().reset_index().rename(columns={'user':'item_count'})
    data = pd.merge(data, p, how='left',on='item')
    data = data[data['item_count'] >= 10].drop(['item_count'], axis=1) # 10번 이상 평가된 item(영화)만 남김

    # 2. Remap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1))) # zip으로 튜플을 만들고, dict으로 변환
    item_map[-1] = 0 # 예외 케이스는 0으로
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x]) # remapping함

    # 3. Remap user ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(user_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # 4. Get user session
    data = data.sort_values(by=['user','timestamp']).reset_index(drop=True) # user 별로 timestamp 순으로 평가 데이터 정렬

    user_session = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    # 5. train, test split
    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    items_user_clicked = {}


    for index, row in user_session.iterrows():
        user = row['user']
        items = row['item_list']

        # (1) test data
        # test data split
        test_item = items[-1 * len_pred:] # 뒤에서 len_pred 만큼 추출
        test_seq = items[-1 * (len_pred + len_seq) : -1 * len_pred]
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_item)

        # (2) train data
        # train_item
        train_item = items[:-1 * len_pred] # items 리스트에서 마지막 len_pred개를 제외한 모든 항목

        # items_user_clicked
        items_user_clicked[user] = train_item

        for i in range(len_seq, len(train_item) - len_tag + 1):
            item = train_item[i : i + len_tag]
            seq = train_item[max(0, i - len_seq) : i]

            train_users.append(user)
            train_seqs.append(seq)
            train_targets.append(item)

    # train, test df 생성
    d_train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    d_test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    d_info= (all_user_count, all_item_count, items_user_clicked, user_map, item_map)

    return d_train, d_test, d_info


if __name__ == '__main__':

    # file_path = './datasets/users.dat'
    file_path = './datasets/u.data'
    names = ['user', 'item', 'rating', 'timestamp']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)
    
    d_train, d_test, d_info = make_datasets(data, 5, 3, 2)
    # print(d_train.head())
    



'''

src data

196	242	3	881250949
186	302	3	891717742
22	377	1	878887116
244	51	2	880606923
166	346	1	886397596
298	474	4	884182806
115	265	2	881171488
253	465	5	891628467
305	451	3	886324817
6	86	3	883603013
62	257	2	879372434
286	1014	5	879781125
200	222	5	876042340
210	40	3	891035994
224	29	3	888104457



train_data
                          seq            target  user
0     [1, 290, 492, 381, 752]    [467, 523, 11]     1
1   [290, 492, 381, 752, 467]    [523, 11, 673]     1
2   [492, 381, 752, 467, 523]   [11, 673, 1046]     1
3    [381, 752, 467, 523, 11]  [673, 1046, 650]     1
4    [752, 467, 523, 11, 673]  [1046, 650, 378]     1
5   [467, 523, 11, 673, 1046]   [650, 378, 180]     1
6   [523, 11, 673, 1046, 650]   [378, 180, 390]     1
7   [11, 673, 1046, 650, 378]   [180, 390, 666]     1
8  [673, 1046, 650, 378, 180]   [390, 666, 513]     1
9  [1046, 650, 378, 180, 390]   [666, 513, 432]     1

test_data
                            seq       target  user
0    [633, 657, 1007, 948, 364]  [522, 0, 0]     1
1       [26, 247, 49, 531, 146]   [32, 0, 0]     2
2      [459, 477, 369, 770, 15]  [306, 0, 0]     3
3  [1093, 946, 1101, 690, 1211]  [526, 0, 0]     4
4     [732, 266, 669, 188, 253]  [986, 0, 0]     5
5      [410, 446, 104, 782, 96]   [26, 0, 0]     6
6     [146, 817, 536, 694, 186]  [525, 0, 0]     7
7      [395, 669, 281, 289, 98]  [731, 0, 0]     8
8     [588, 671, 369, 292, 304]  [250, 0, 0]     9
9        [472, 222, 82, 716, 8]  [131, 0, 0]    10


'''