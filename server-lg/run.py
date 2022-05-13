#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
           run.py


'''
__author__  = 'Ferriol Pey Comas [ferriol73pey@gmail.com]'
__version__ = 'v1.0'
__date__    = '05/05/2022'

import warnings
warnings.filterwarnings('ignore')

from mdl import get_default_net
from extended_config import cfg as conf
import PIL
import spacy
import numpy as np
from evaluator import Evaluator
import mdl
import torch
import cv2


MODEL_PATH = 'model.pth'
phrase_len=50

def pil2tensor(image, dtype: np.dtype):
    "Convert PIL style image array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))

def collater(batch):
    qlens = torch.Tensor([i['qlens'] for i in batch])
    max_qlen = int(qlens.max().item())
    out_dict = {}
    for k in batch[0]:
        out_dict[k] = torch.stack([b[k] for b in batch]).float()
    out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

    return out_dict

cfg = conf
cfg.mdl_to_use = 'ssd_vgg_t'
cfg.ds_to_use = 'refclef'
cfg.num_gpus = 1
cfg.bs=16
cfg.nw=4
device = torch.device(cfg.device)
ratios = eval(cfg['ratios'], {})
scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))

n_anchors = len(ratios) * len(scales)
cfg.mdl_to_use = 'retina'

model = get_default_net(num_anchors=n_anchors,cfg=cfg)
pretrained_model = torch.load(MODEL_PATH)
state_dict = pretrained_model['model_state_dict']
loaded_state_dict = pretrained_model['model_state_dict']
loaded_state_dict2 = {k.split('module.',1)[1]: v for k,v in loaded_state_dict.items()}

model.load_state_dict(loaded_state_dict2, strict=True)

model.phase='test'

nlp = spacy.load('en_core_web_md')

###############################################################################
#
#                                  FLASK Server
#
###############################################################################



from flask import Flask, request, jsonify # for the flask server
import base64
from io import BytesIO
import json

app = Flask(__name__)


def find(img,query):
    '''

    This function tries to convert into text some input audio.

    ==========
    Parameters
    ==========

       img     : image data
            It conains the data of the image where to find the object.
       query   : string
            It is the string that contains the description of the object.
    =======
    RETURNS
    =======

       result : json
           It returns a rectangle containing the object and a confidence score.
    '''
    h, w = img.height, img.width
    q_chosen = query.strip()
    qtmp = nlp(str(q_chosen))
    if len(qtmp) == 0:
        raise NotImplementedError
    qlen = len(qtmp)
    q_chosen = q_chosen + ' PD'*(phrase_len - qlen)
    q_chosen_emb = nlp(q_chosen)
    if not len(q_chosen_emb) == phrase_len:
        q_chosen_emb = q_chosen_emb[:phrase_len]
    q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
    img = img.resize((cfg.resize_img[0], cfg.resize_img[1]))
    target = np.array([
        0 / h, 0 / w,
        0 / h, 0 / w
    ])

    img = pil2tensor(img, np.float_).float().div_(255)

    target = np.array([ 0, 0, 0, 0])
    out = {
        'img': img, # torch.rand(*img.shape), #
        'qvec': torch.from_numpy(q_chosen_emb_vecs), #torch.rand(*q_chosen_emb_vecs.shape),
        'qlens': torch.tensor(qlen),
        'annot': torch.from_numpy(target).float(),
        'img_size': torch.tensor([h, w]),
        'idxs': torch.tensor(qlen),
    }


    col = collater([out])
    evl = Evaluator(ratios, scales, cfg)

    model.to(device)
    model.eval()
    for c in col.keys():
        col[c] = col[c].to(device)
    mdl_out = model(col)
    predictions = evl(mdl_out, col)

    pred_boxes = predictions['pred_boxes']
    pred_scores = predictions['pred_scores']

    box = pred_boxes.data.cpu().numpy()[0]
    score = pred_scores.data.cpu().numpy()[0]
    boxx = [int(box[0]),int(box[1]),int(box[2]),int(box[3]) ]
    return jsonify({"box":boxx, "score":float(score)})



@app.route("/", methods=["GET", "POST"])
def index():
    '''
    This function processes all the requests done to the url
    "/api/recognize/*" in order to return the position of the
    required object.
    ==========
    Parameters
    ==========
       uuid :  int
            It conains the uuid that the client selected and sent using
            the url.
    =======
    RETURNS
    =======
       result : json
           It returns the recognized sentence and the confidence, start
           time and end time for each recognized word.
    '''
    json_data = request.get_json() #Get the POSTed json
    dict_data = json.loads(json_data) #Convert json to dictionary

    img = dict_data["img"] #Take out base64# str
    img = base64.b64decode(img) #Convert image data converted to base64 to original binary data# bytes
    img = BytesIO(img) # _io.Converted to be handled by BytesIO pillow
    #Converts the image
    img = PIL.Image.open(img).convert('RGB')
    qry = dict_data["query"]
    return find(img,qry)          # returns the result of the recognition using the recognize function

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=False)#True)         # starts the application
