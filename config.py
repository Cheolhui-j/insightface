from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
import os

def get_config(training = True):
    conf = edict()

    #===============================================================================
    # PATH
    #===============================================================================
    # Data Path
    conf.data_path = '/mnt/cheolhui/Training_2/data'
    conf.data_mode = 'emore'
    conf.vgg_folder = os.path.join(conf.data_path, 'faces_vgg_112x112')
    conf.ms1m_folder = os.path.join(conf.data_path, 'faces_ms1m_112x112')
    conf.emore_folder = os.path.join(conf.data_path, 'faces_emore')

    # Save Path
    conf.work_path = '/mnt/cheolhui/all_pair/work_space/mv' # cutoff
    #conf.work_path = '/mnt/cheolhui/all_pair/work_space/r50_cosine' # r50
    #conf.work_path = '/mnt/cheolhui/all_pair/work_space/r100_cosine' # r100
    #conf.work_path = '/mnt/cheolhui/all_pair/work_space/translation' # r100
    conf.model_path = os.path.join(conf.work_path, 'models')
    conf.log_path = os.path.join(conf.work_path, 'logs')
    conf.save_path = os.path.join(conf.work_path, 'save')

    #===============================================================================
    # Training Env
    #===============================================================================
    conf.gpu_ids = [6]
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #===============================================================================
    # Model & Configs
    #===============================================================================
    # Trainint

    # ir, ir_se : 50, 100, 152
    # mobilefacenet, 
    # resnet : 18, 34, 50, 101, 152
    # lightcnn: 9, 29
    conf.net_mode = 'resnet'
    conf.net_depth = 101
    #conf.net_depth = 101
    conf.use_ADAM = True
    
    conf.batch_size = 512 # irse net depth 50
    #conf.batch_size = 200

    conf.epochs = 2000
    #conf.lr = 1e-1 # Default init learning rate
    # conf.lr = 1e-4 # resnet_test1: failed
    
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    
    conf.drop_ratio = 0.6
    
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
    
#--------------------Training Config ------------------------    
    if training:        
        # conf.log_path = conf.work_path/'log'
        # conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-1
        conf.milestones = [7,14,20]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path+'/'+'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
