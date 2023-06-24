import tensorflow as tf
import math


class FPMC(object):

    def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag):

        self.emb_size = emb_size # 임베딩 크기
        self.user_count = num_usr # 사용자 수
        self.item_count = num_item # 아이템 수
        self.len_Seq = len_Seq # 시퀀스의 길이
        self.len_Tag = len_Tag, # 태그의 길이

        # 임베딩 초기화. 평균이 0이고 표준편차가 0.1인 정규 분포에서 임의의 값을 생성하여 초기화에 사용함.
        self.init = tf.random_normal_initializer(0, 0.1)

        # 입력 데이터 생성
        self.input_Seq = tf.placeholder(tf.int32, [None, self.len_Seq]) #[B,T] # 시퀀스 데이터의 입력으로 [배치 크기, 시퀀스 길이]임
        self.input_Usr = tf.placeholder(tf.int32, [None]) #[B] # 사용자 데이터 입력
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F] # neg 태그 데이터 입력
        self.input_PosT = tf.placeholder(tf.int32, [None, None]) #[B] # pos 태그 데이터 입력

        # loss 값
        self.loss = self.build_model(self.input_Seq, self.input_Usr, self.input_NegT, self.input_PosT)

    # PMF와 MF 계산을 수행하는 메서드
    def PMFC(self, Vui, Viu, Vil, Vli):
        # Vui, Viu, Vil, Vli는 임베딩 텐서
        '''
        :param Vui: [b,1,e]
        :param Viu: [b,S,e]
        :param Vil: [b,S,e]
        :param Vli: [b,L,e]
        :return:
        '''

        # MF
        mf = tf.matmul(Vui, tf.transpose(Viu,[0,2,1])) # [b(배치크기), 1, S(시퀀스길이)] # Vui와 Viu 행렬곱을 계산해서 사용자와 시퀀스 간의 상호작용을 모델링함
        mf = tf.squeeze(mf,1) # [b, S] # 차원이 1인 축을 제거하고, 사용자와 시퀀스 간 상호작용 텐서 mf를 얻음.

        #PMF
        pmf = tf.matmul(Vil, tf.transpose(Vli,[0,2,1])) # [b(배치크기), S(시퀀스길이), L(아이템개수)] # 시퀀스 내 아이템 간의 상호작용을 모델링함.
        pmf = tf.reduce_mean(pmf, -1) #[b, S, 1] # 텐서의 마지막 차원을 제거하고, 평균 값을 계산해 -> 시퀀스 내 아이템 간 상호작용의 평균을 얻는다.
        
        # MF와 PMF 결합
        x = pmf + mf #[B, S]

        return  x

    # 손실함수: 예측값 X_uti와 X_utj 간의 차이에 대한 로그-시그모이드 손실을 계산하여 평균화한 값으로 평균 손실을 얻는다.
    # (4. mean_loss 반환(3. 평균 손실 계산((2. 로그-시그모이드 손실 계산(1. 모델의 예측값 차이 계산)))))
    def loss_function(self, X_uti, X_utj):
        return - 1* tf.reduce_mean(tf.log(tf.sigmoid(tf.squeeze(X_uti - X_utj))),-1)

    # 모델 구축
    def build_model(self, in_Seq, in_Usr, in_Neg, in_Pos):
        # in_Seq: 입력 시퀀스
        # in_Usr: 사용자
        # in_Neg: 부정적인 아이템
        # in_Pos: 긍정적인 아이템

        self.UI_emb = tf.get_variable("UI_emb", [self.user_count, self.emb_size],initializer=self.init)  #[N,e] # 사용자
        self.IU_emb = tf.get_variable("IU_emb", [self.item_count, self.emb_size],initializer=self.init)  # [N,e] # 아이템
        self.LI_emb = tf.get_variable("LI_emb", [self.item_count, self.emb_size],initializer=self.init)  #[N,e] # 시퀀스 아이템
        self.IL_emb = tf.get_variable("IL_emb", [self.item_count, self.emb_size],initializer=self.init)  # [N,e] # 시퀀스 태그

        ui = tf.nn.embedding_lookup(self.UI_emb, in_Usr) #[b,1,1] # UI_emb에서 in_Usr에 해당하는 임베딩을 찾아 ui에 할당함.
        ui = tf.expand_dims(ui, 1)
        seq = tf.nn.embedding_lookup(self.LI_emb, in_Seq) #[b,l,1] # LI_emb에서 in_Seq에 해당하는 임베딩을 찾아 seq에 할당함.

        pos_iu = tf.nn.embedding_lookup(self.IU_emb,in_Pos) #[b,1,1] # IU_emb에서 in_Pos에 해당하는 임베딩을 찾아 pos_iu에 할당
        pos_il = tf.nn.embedding_lookup(self.IL_emb, in_Pos)#[b,1,1] # IL_emb에서 in_Pos에 해당하는 임베딩을 찾아 pos_il에 할당
        pos_score = self.PMFC(ui, pos_iu, pos_il, seq)

        # neg 아이템에 대해서 PMF 계산
        neg_iu = tf.nn.embedding_lookup(self.IU_emb, in_Neg)
        neg_il = tf.nn.embedding_lookup(self.IL_emb, in_Neg)
        neg_score = self.PMFC(ui, neg_iu, neg_il, seq)
        
        # 손실함수 계산
        loss = self.loss_function(pos_score, neg_score)

        return loss
    
    # 예측 점수 반환
    def predict(self):

        # 텐서 임베딩 조회
        ui = tf.nn.embedding_lookup(self.UI_emb, self.input_Usr)  #[B,1,1]
        ui = tf.expand_dims(ui, 1)
        seq = tf.nn.embedding_lookup(self.LI_emb, self.input_Seq) #[B,T,1]
        pos_iu = tf.tile(tf.expand_dims(self.IU_emb,0),[tf.shape(self.input_Usr)[0] ,1, 1])
        pos_il = tf.tile(tf.expand_dims(self.IL_emb, 0),[tf.shape(self.input_Usr)[0], 1, 1])

        # 예측 점수 계산
        score = self.PMFC(ui, pos_iu, pos_il, seq)

        return score