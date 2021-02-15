from config import get_config
from Learner import face_learner
import numpy as np
from data.data_pipe import get_val_pair
import os
from torchvision import transforms as trans
import cv2
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
Hist_size = 10001
image_list_size = 13233

def checkissame(src, dst):

    name1 = '_'.join(src.split('_')[:-1])
    name2 = '_'.join(dst.split('_')[:-1])

    if name1 == name2:
        issame = True

    else:
        issame = False

    return issame


def all_pair_lfw(learner_path, list_path, out_text):

    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load config
    conf = get_config()

    # Load model
    learner = face_learner(conf)
    learner.load_state(conf, learner_path, model_only=True) # cutoff
    learner.model.eval()

    # Load file
    f_src = open(list_path, 'r')
    data = f_src.readlines()

    # Load image from file list
    img_root = '/mnt/cheolhui/all_pair/lfw_align_lib/'
    img_total = torch.zeros([image_list_size, 3, 112, 112])

    for i, image_name in enumerate(data):
            lis = '_'.join(image_name.split('_')[:-1])
            img_path = img_root + lis + '/' + image_name.rstrip('\n')
            image = cv2.imread(img_path)
            img_total[i] = test_transform(image)

    embeddings = torch.zeros([image_list_size, conf.embedding_size])

    # extract features from images
    # if all image implement embedding, memory expload. so split image 
    idx = 0
    with torch.no_grad():
        while idx + conf.batch_size <= image_list_size:
            batch = img_total[idx:idx + conf.batch_size]
            embeddings[idx:idx + conf.batch_size] = learner.model(batch.to(conf.device))
            idx += conf.batch_size
        if idx < image_list_size:
            batch = torch.tensor(img_total[idx:])
            embeddings[idx:] = learner.model(batch.to(conf.device))


    # create histogram
    HIST_Genuine = np.zeros(Hist_size)
    HIST_imposter = np.zeros(Hist_size)

    # calculate L2 - distance between two embeddings in order of all images
    # fill each histogram according to the distances
    for i , embedding1 in enumerate(embeddings):

        print("Matching... {}".format(data[i]))

        for j, embedding2 in enumerate(embeddings):

            dist = (embedding1 - embedding2).norm(p=2)
            dist = dist.cpu().detach().numpy()
            dist = 1.0 - (float(dist)/2)

            sim_bin = min(dist * Hist_size - 1, Hist_size - 1)

            if checkissame(data[i], data[j]):
                HIST_Genuine[int(sim_bin)] = HIST_Genuine[int(sim_bin)] + 1
            else :
                HIST_imposter[int(sim_bin)] = HIST_imposter[int(sim_bin)] + 1

    # write hist 
    with open(out_text, "wt") as f:
         for i in range(Hist_size):
             f.writelines("{}\t{}\n".format(HIST_Genuine[i], HIST_imposter[i]))

    print('Process done')



if __name__ == "__main__":
    learner_path  = 'resnet_101_accuracy:0.9714_step:367571_None.pth'
    list_path = './lfw_list_suprema.txt'
    out_text = "./hist_mv.txt"
    all_pair_lfw(learner_path, list_path)