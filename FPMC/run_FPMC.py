import sys
import os
from tqdm import tqdm
import pandas as pd
import argparse
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import tensorflow as tf
import numpy as np
from model_FPMC import FPMC
from make_datasets import make_datasets
from data_input import DataIterator
from evaluation import SortItemsbyScore,Metric_HR,Metric_MRR


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--num_epochs', type=int, default=30) # 에포크 수
    parser.add_argument('--emb_size', type=int, default=50) # 아이템 임베딩의 크기
    parser.add_argument('--len_Seq', type=int, default=5) # 입력 시퀀스
    parser.add_argument('--len_Tag', type=int, default=1) # 태그 길이
    parser.add_argument('--len_Pred', type=int, default=1) # 학습 배치 
    parser.add_argument('--neg_sample', type=int, default=10) # neg sample 수
    parser.add_argument('--batch_size', type=int, default=50) # 학습 배치 크기
    parser.add_argument('--learning_rate', type=float, default=1e-2) # 학습률
    parser.add_argument('--l2_lambda', type=float, default=1e-6) # L2 정규화 lambda값
    return parser.parse_args()

if __name__ == '__main__':

    # Get Params
    args = parse_args()
    len_Seq = args.len_Seq
    len_Tag = args.len_Tag
    len_Pred = args.len_Pred
    batch_size = args.batch_size
    emb_size = args.emb_size
    neg_sample = args.neg_sample

    l2_lambda = args.l2_lambda
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # make datasets

    print('==> make datasets <==')
    file_path = './datasets/u.data'
    names = ['user', 'item', 'rateing', 'timestamp']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)
    d_train, d_test, d_info = make_datasets(data, len_Seq, len_Tag, len_Pred)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    trainIterator = DataIterator('train',d_train, batch_size, neg_sample,
                                 all_items, items_usr_clicked, shuffle=True)
    testIterator = DataIterator('test',d_test, batch_size,  shuffle=False)

    # Define Model

    model = FPMC(emb_size, num_usr, num_item, len_Seq, len_Tag)
    loss = model.loss
    input_Seq = model.input_Seq
    input_Usr = model.input_Usr # [B]
    input_NegT = model.input_NegT
    input_PosT = model.input_PosT
    score_pred = model.predict()

    # Define Optimizer

    global_step = tf.Variable(0, trainable=False) # 학습 중 현재까지의 전체 스텝 수 추적하는 데 사용되는 변수
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 배치 정규화와 관련된 업데이트 연산을 가져옴
    with tf.control_dependencies(update_ops): # 파라미터 업데이트 연산 전에 배치 정규화 연산이 먼저 실행되도록 함
        optimizer = tf.train.AdamOptimizer(learning_rate) # Adam 옵티마이저를 생성
        tvars = tf.trainable_variables() # 학습 가능한 변수를 가져옴
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5) # 손실에 대한 그래디언트 계산
        grads_and_vars = tuple(zip(grads, tvars)) # 그래디언트와 변수를 쌍으로 묶어서 튜플 형태로 지정
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # 그래디언트를 적용하여 파라미터를 업데이트 하는 학습 연산 정의

    # Training and test for every epoch

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            #train
            cost_list = []
            for train_input in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                feed_dict = {input_Usr: batch_usr, input_Seq: batch_seq,
                            input_PosT: batch_pos, input_NegT: batch_neg}
                _, step, cost= sess.run([train_op, global_step, loss],feed_dict)
                cost_list += list(cost)
            mean_cost = np.mean(cost_list)
            #saver.save(sess, FLAGS.save_path)

            # test

            pred_list = []
            next_list = []
            user_list = []

            for test_input in testIterator:
                batch_usr, batch_seq, batch_pos, batch_neg = test_input
                feed_dict = {input_Usr: batch_usr, input_Seq: batch_seq}
                pred = sess.run(score_pred, feed_dict)  # , options=options, run_metadata=run_metadata)

                pred_list += pred.tolist()
                next_list += list(batch_pos)
                user_list += list(batch_usr)

            sorted_items,sorted_score = SortItemsbyScore(all_items,pred_list,reverse=True,remove_hist=True
                                                   ,usr=user_list,usrclick=items_usr_clicked)
            #
            hr50 = Metric_HR(50, next_list, sorted_items)
            Mrr = Metric_MRR(next_list, sorted_items)
            print(" epoch {}, mean_loss{:g}, test HR@50: {:g} MRR: {:g}"
                  .format(epoch + 1, mean_cost, hr50, Mrr))