'''
DataIterator는 훈련 데이터, 테스트 데이터를 배치 단위로 제공하는 역할을 함
train_iterator = DataIterator('train', d_train, batch_size, neg_sample, all_items, items_user_clicked, shuffle=True)
'''


import pandas as pd
import random
import pickle
from tqdm import tqdm
from make_datasets import make_datasets


class DataIterator:
    
    # __init__ 메서드: 클래스의 인스턴스를 초기화함. 데이터를 받아오고 필요한 매개변수들을 설정한다.
    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_user_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_user_clicked = items_user_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    
    # __iter__ 메서드: 이터레이터를 반환함. 이 메서드를 구현하여 객체를 반복가능하게 만들 수 있다.
    def __iter__(self):
        return self


    # reset 메서드: 이터레이터를 초기화하고 데이터를 셔플함.
    def reset(self):
        self.idx = 0 # 이터레이터의 인덱스를 0으로 초기화해서, 데이터를 처음부터 순회하게 함.
        if self.shuffle: # shuffle 매개변수가 True일 경우 데이터를 섞는다.
            self.data= self.data.sample(frac=1).reset_index(drop=True) # frac=1 전체 데이터를 무작위로 섞어서 반환함.
            self.seed = self.seed + 1
            random.seed(self.seed)


    # __next__ 메서드: 데이터를 읽고 다음 배치 단위를 반환하며, 훈련 데이터인 경우 negative sample도 생성한다.
    def __next__(self):

        if self.idx >= self.datasize: # 현재 인덱스(self.idx)가 데이터의 크기보다 크거나 같으면,
            self.reset() # 데이터를 초기화하고
            raise StopIteration # 예외를 발생시켜 반복을 종료한다.

        nums = self.batch_size 
        if self.datasize - self.idx < self.batch_size: # (데이터의 크기 - 현재 인덱스) < (배치의 크기) 라면,
            nums  = self.datasize - self.idx # nums는 (데이터의 크기 - 현재 인덱스)임.

        cur = self.data.iloc[self.idx : self.idx + nums] # 현재 배치에 해당하는 데이터를 슬라이싱하여 cur에 할당함. cur은 현재 배치 데이터임.

        batch_user = cur['user'].values # 현재 배치의 'user' 열 데이터를 배열 형태tem_usr_clicked로 할당함.

        batch_seq = []
        for seq in cur['seq'].values: # 현재 배치의 'seq' 열 데이터를 배열 형태로 할당함.
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values: # 현재 배치의 'target' 열 데이터를 배열 형태로 할당함. positive item.
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train': # negative sampling은 mode가 train인 경우만 수행됨.
            for u in cur['user']:
                user_item_set = set(self.all_items) - set(self.item_user_clicked[u]) # 현재 배치의 'user' 값을 기반으로 해당 사용자가 클릭하지 않은 아이템 집합을 구함.
                batch_neg.append(random.sample(user_item_set, self.neg_count)) # user가 클릭하지 않은 아이템 집합에서 neg_count 개수만큼 negative 예제를 무작위로 샘플링하여 batch_neg에 추가함.

        self.idx += self.batch_size # 현재 인덱스를 배치 크기만큼 증가시킴

        return (batch_user, batch_seq, batch_pos, batch_neg)

if __name__ == '__main__':
    
    file_path = './datasets/users.dat'
    names = ['user', 'item', 'rating', 'timestamp']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)
    
    d_train, d_test, d_info = make_datasets(data, 5, 3, 4)
    num_user, num_item, items_user_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    train_iterator = DataIterator('train', d_train, 21, 5,
                                 all_items, items_user_clicked, shuffle=True)
    for epoch in range(6):
        for data in tqdm(train_iterator,desc='epoch {}'.format(epoch),total=train_iterator.total_batch):
            batch_user, batch_seq, batch_pos, batch_neg = data