from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import torch
import random
import argparse
import cv2
import time
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from config import get_config
from easydict import EasyDict as edict

from Learner import face_learner

from data.data_pipe import load_bin

def load_property(data_dir):
  prop = edict()
  for line in open(os.path.join(data_dir, 'property')):
    vec = line.strip().split(',')
    assert len(vec)==3
    prop.num_classes = int(vec[0])
    prop.image_size = [int(vec[1]), int(vec[2])]
  return prop

def get_embedding(args, imgrec, id, image_size, model, conf):
  s = imgrec.read_idx(id)
  header, _ = mx.recordio.unpack(s)
  ocontents = []
  for idx in range(int(header.label[0]), int(header.label[1])):
    #print('idx', idx)
    s = imgrec.read_idx(idx)
    ocontents.append(s)
  embeddings = None
  #print(len(ocontents))
  ba = 0
  while True:
    bb = min(ba+args.batch_size, len(ocontents))
    if ba>=bb:
      break
    _batch_size = bb-ba
    #_batch_size2 = max(_batch_size, args.ctx_num)
    data = torch.zeros( (_batch_size,3, image_size[0], image_size[1]) )
    #label = nd.zeros( (_batch_size2,) )
    count = bb-ba
    ii=0
    for i in range(ba, bb):
      header, img = mx.recordio.unpack(ocontents[i])
      #print(header.label.shape, header.label)
      img = mx.image.imdecode(img)
      img = nd.transpose(img, axes=(2, 0, 1))
      data[ii][:] = torch.tensor(img.asnumpy())
      #label[ii][:] = header.label
      ii+=1
    while ii<_batch_size:
      data[ii][:] = data[0][:]
      #label[ii][:] = label[0][:]
      ii+=1
    net_out = model(data.to(conf.device)).cpu()
    if embeddings is None:
      embeddings = np.zeros( (len(ocontents), net_out.shape[1]))
    embeddings[ba:bb,:] = net_out[0:_batch_size,:].detach().numpy()
    ba = bb
  embeddings = sklearn.preprocessing.normalize(embeddings)
  embedding = np.mean(embeddings, axis=0, keepdims=True)
  #embedding = embeddings[random_num,np.newaxis]
  embedding = sklearn.preprocessing.normalize(embedding).flatten()
  return embedding

def main(args):
  start_time = time.time()
  merge_datasets = args.merge.split(',')
  # load dataset1 class_num & img_size
  prop = load_property(merge_datasets[0])
  image_size = prop.image_size
  print('image_size', image_size)
  learner = None
  conf = get_config()
  # load model
  if args.model:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('loading', args.model)
    learner = face_learner(conf, train=False)
    learner.load_state(conf, '{}.pth'.format(args.model), model_only=True) # r100
    learner.model.eval()
  else:
    print('model is empty')
    assert args.t==0.0
  rec_list = []
  # read rec
  for ds in merge_datasets:
    path_imgrec = os.path.join(ds, 'train.rec')
    path_imgidx = os.path.join(ds, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    rec_list.append(imgrec)

  id_list_map = {}
  all_id_list = []
  #
  for ds_id in range(len(rec_list)):
    id_list = []
    imgrec = rec_list[ds_id]
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag>0
    print('header0 label', header.label)
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    pp=0
    for identity in seq_identity:
      pp+=1
      if pp%10==0:
        print('processing id', pp)
      if learner is not None:
        embedding = get_embedding(args, imgrec, identity, image_size, learner.model, conf)
      else:
        embedding = None
      #print(embedding.shape)
      id_list.append( [ds_id, identity, embedding] )
    id_list_map[ds_id] = id_list
    if ds_id==0 or learner is None:
      all_id_list += id_list
      print(ds_id, len(id_list))
    else:
      X = []
      #data_hist = []
      for id_item in all_id_list:
        X.append(id_item[2])
      X = np.array(X)
      i = 0
      for i in range(len(id_list)):
        id_item = id_list[i]
        y = id_item[2]
        sim = np.dot(X, y.T) #memory error
        print(i)
        #data_hist.append(sim)
        idx = np.where(sim>=args.t)[0]
        if len(idx)>0:
          continue
        all_id_list.append(id_item)
        i += 1
      #rng_hist = np.arange(-1, 1.05, 0.05)
      #data_hist = np.asarray(data_hist)
      #res_hist, res_bins, _ = plt.hist(data_hist, rng_hist, rwidth = 0.8)
      #print(res_hist)
      #plt.show()

  if not os.path.exists(args.output):
    os.makedirs(args.output)
  writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'train.idx'), os.path.join(args.output, 'train.rec'), 'w')
  idx = 1
  identities = []
  nlabel = -1
  for id_item in all_id_list:
    if id_item[1]<0:
      continue
    nlabel+=1
    ds_id = id_item[0]
    imgrec = rec_list[ds_id]
    id = id_item[1]
    s = imgrec.read_idx(id)
    header, _ = mx.recordio.unpack(s)
    a, b = int(header.label[0]), int(header.label[1])
    identities.append( (idx, idx+b-a) )
    for _idx in range(a,b):
      s = imgrec.read_idx(_idx)
      _header, _content = mx.recordio.unpack(s)
      nheader = mx.recordio.IRHeader(0, nlabel, idx, 0)
      s = mx.recordio.pack(nheader, _content)
      writer.write_idx(idx, s)
      idx+=1
  id_idx = idx
  for id_label in identities:
    _header = mx.recordio.IRHeader(1, id_label, idx, 0)
    s = mx.recordio.pack(_header, b'')
    writer.write_idx(idx, s)
    idx+=1
  _header = mx.recordio.IRHeader(1, (id_idx, idx), 0, 0)
  s = mx.recordio.pack(_header, b'')
  writer.write_idx(0, s)
  with open(os.path.join(args.output, 'property'), 'w') as f:
    f.write("%d,%d,%d"%(len(identities), image_size[0], image_size[1]))
  print('time : {}'.format(time.time()-start_time))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do dataset merge')
  # general
  parser.add_argument('--merge', default='../faces_umd,../faces_umd', type=str, help='')
  parser.add_argument('--output', default='./merge', type=str, help='')
  parser.add_argument('--model', default='r100_cosine', help='path to load model.')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  parser.add_argument('--t', default=0.65, type=float, help='threshold to divide the image to be added based on image similarity')
  args = parser.parse_args()
  main(args)

