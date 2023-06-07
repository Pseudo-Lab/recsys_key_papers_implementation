import math

def hit_at_k(ranked_tuples, k):
    '''
    Checks if the pos interaction occured in the top k scores
    '''
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            return 1
    return 0

def ndcg_at_k(ranked_tuples, k):
    '''
    Article on ndcg: http://ethen8181.github.io/machine-learning/recsys/2_implicit.html
    ndcg_k = DCG_k / IDCG_k
    Say i represents index of or tag=1 in the top k, then since only one contribution to summation
    DCG_k = rel_i / log(i+1)
    IDCG_k = rel_i / log(1+1) since the first item is best rank
    ndcg_k = log(2) / log(i+1)
    Note we use log(2) / log(i+2) since indexing from 0
    '''
    for i,(score, tag) in enumerate(ranked_tuples[:k]):
        if tag == 1:
            return math.log(2) / math.log(i + 2)
    return 0
