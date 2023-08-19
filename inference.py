import argparse
import json
from typing import List

import torch

from SASRec.model import SASRec


parser = argparse.ArgumentParser()
args = parser.parse_args()
with open('sasrec_model/args.txt', 'r') as f:
    args.__dict__ = json.load(f)

sasrec_model = SASRec(args.usernum, args.itemnum, args)
sasrec_model.load_state_dict(torch.load('sasrec_model/SASRec_epoch_199.pth'))

def sasrec_inference(past_interactions:List[int]):

    return past_interactions[-1] + 1