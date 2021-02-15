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
    conf.work_path = '/mnt/cheolhui/all_pair/work_space/mv' 
    conf.model_path = os.path.join(conf.work_path, 'models')
    conf.log_path = os.path.join(conf.work_path, 'logs')

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
    # resnet : 18, 34, 50, 101, 152
    # lightcnn: 9, 29
    conf.net_mode = 'resnet'
    conf.net_depth = 101
    #conf.net_depth = 101
    conf.use_ADAM = True
    
    conf.batch_size = 512 # irse net depth 50

    conf.epochs = 2000
    
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    
    conf.drop_ratio = 0.6
    
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
         
    conf.lr = 1e-1
    conf.milestones = [7,14,20]
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 3
    conf.ce_loss = CrossEntropyLoss() 
    conf.cosine_lr = False  

    return conf
