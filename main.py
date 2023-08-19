import json
import threading
import time
from queue import Empty, Queue

import numpy as np
import torch
from flask import Flask, request as flask_request
from flask_restx import Namespace, Api, Resource, fields
from SASRec.model import SASRec

app = Flask(__name__)
api = Api(app, version='1.0', title='API 문서 - 배치 프로세싱', description='과연', doc="/")
ns = api.namespace('SASRec', description='배치 프로세싱 테스트 ns')
model_goods = api.model('used_items_seqs', {
    'seqs': fields.String(readOnly=True, required=True, description='사용한 아이템 시퀀스', help='상품번호는 필수', example='[1, 2, 3]')
})

BATCH_SIZE = 50
BATCH_TIMEOUT = 0.5
CHECK_INTERVAL = 0.01

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
with open('SASRec/sasrec_model/args.txt', 'r') as f:
    args = json.load(f)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
args = dotdict(args)
print(args)
print(args.itemnum)
args.itemnum = int(args.itemnum)
args.usernum = int(args.usernum)
sasrec_model = SASRec(args.usernum, args.itemnum, args)
sasrec_model.load_state_dict(torch.load('SASRec/sasrec_model/SASRec_epoch_199.pth'))

requests_queue = Queue()


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) > BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        print(f"@@@ requests_queue : {requests_queue.queue}")
        print(f"@@@ len(requests_batch) : {len(requests_batch)}")
        batch_inputs = np.array([request['input'] for request in requests_batch])
        item_indices = [list(range(1, 100)) for _ in range(len(batch_inputs))]
        batch_outputs = sasrec_model.predict(log_seqs=batch_inputs, item_indices=item_indices)
        print(f"@@@ batch_outputs.size() : {batch_outputs.size()}")
        for request, output in zip(requests_batch, batch_outputs):
            request['output'] = output


threading.Thread(target=handle_requests_by_batch).start()


@ns.route('/predict')
class Predict(Resource):

    @ns.expect(model_goods)
    def post(self):
        # print(f"!!! flask_request : {flask_request}")
        print(f"!!! type(flask_request) : {type(flask_request)}")
        print(f"!!! flask_request.json : {flask_request.json}")
        received_input = np.array(eval(flask_request.json['seqs']))
        request = {'input': received_input, 'time': time.time()}
        requests_queue.put(request)

        while 'output' not in request:
            time.sleep(CHECK_INTERVAL)

        return {'predictions': request['output'].tolist()}


if __name__ == '__main__':
    app.run()
