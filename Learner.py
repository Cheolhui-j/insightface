from data.data_pipe import de_preprocess, get_train_loader, get_val_data

from model.model_ir import IRNet
from model.model_resnet import ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152
from model.model_lightCNN import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from model.arcface import Arcface, Am_softmax, l2_norm
from verifacation import evaluate
import os
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import gen_plot, hflip_batch, separate_bn_paras,\
     make_dir #, CosineAnnealingWarmUpRestarts
from PIL import Image
from torchvision import transforms as trans

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class face_learner(object):
    def __init__(self, conf, train=True):
        make_dir(conf.work_path)        
        make_dir(conf.model_path)
        make_dir(conf.log_path)

        if conf.gpu_ids:
            assert torch.cuda.is_available(), 'GPU is not avalialble!'
            torch.backends.cudnn.benckmark = True 
            conf.device = torch.device('cuda')
        else:
            conf.device = torch.device('cpu')
        
        self.gpu_ids = conf.gpu_ids

        self.model = None
        self.net_type = '{}_{}'.format(conf.net_mode, conf.net_depth)
        if conf.net_mode == 'ir' or conf.net_mode =='ir_se':
            self.model = IRNet(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        elif conf.net_mode == 'resnet':
            if conf.net_depth == 18:
                self.model = ResNet_18().to(conf.device)
            elif conf.net_depth == 34:
                self.model = ResNet_34().to(conf.device)
            elif conf.net_depth == 50:
                self.model = ResNet_50().to(conf.device)
            elif conf.net_depth == 101:
                self.model = ResNet_101().to(conf.device)
            elif conf.net_depth == 152:
                self.model = ResNet_152().to(conf.device)
            else:
                raise NotImplementedError("Model {}_{} is not implemented".format(
                        conf.net_mode, conf.net_depth))
        elif conf.net_mode == 'lightcnn':
            if conf.net_depth == 9:
                self.model = LightCNN_9Layers(drop_ratio=conf.drop_ratio).to(conf.device)
            elif conf.net_depth == 29:
                self.model = LightCNN_29Layers(drop_ratio=conf.drop_ratio).to(conf.device)
            else:
                raise NotImplementedError("Model {}_{} is not implemented".format(
                        conf.net_mode, conf.net_depth))
        else:
            NotImplementedError("Model {}_{} is not implemented".format(
                        conf.net_mode, conf.net_depth))       

        assert self.model is not None, "Model is NONE!!"

        if train:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_ADAM:
                self.optimizer = optim.Adam([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, betas=(0.9, 0.999))
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)


            # if conf.cosine_lr == True:
            #     self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(conf.data_folder)

    def save_state(self, conf, accuracy, extra=None, model_only=False):

        save_path = conf.model_path
        
        # network = None
        # if isinstance(self.model, torch.nn.DataParallel):
        #     network = self.model.module
        # else: network = self.model

        network = self.model
        assert network is not None

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        ###############################################
        save_name = os.path.join(
            save_path, 'model_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
        torch.save(state_dict, save_name)
        if not model_only:
            # head_module = None
            # if isinstance(self.head, torch.nn.DataParallel):
            #     head_module = self.head.module
            # else: head_module = self.head
            head_module = self.head
            assert head_module is not None
            head_state_dict = head_module.state_dict()
            for key, param in head_state_dict.items():
                head_state_dict[key] = param.cpu()
            save_name = os.path.join(
                save_path, 'head_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
            torch.save(head_state_dict, save_name)

            # optimizer_module = None
            # if isinstance(self.optimizer, torch.nn.DataParallel):
            #     optimizer_module = self.optimizer.module
            # else: optimizer_module = self.optimizer
            optimizer_module = self.optimizer
            assert optimizer_module is not None
            optimizer_state_dict = optimizer_module.state_dict()
            for key, param in head_state_dict.items():
                head_state_dict[key] = param.cpu()
            save_name = os.path.join(
                save_path, 'optimizer_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
            torch.save(optimizer_state_dict, save_name)

            
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):

        save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path+'/'+'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path+'/'+'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path+'/'+'optimizer_{}'.format(fixed_str)))
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, far, frr, eer):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_scalar('{}_FAR'.format(db_name), far, self.step)
        self.writer.add_scalar('{}_FRR'.format(db_name), frr, self.step)
        self.writer.add_scalar('{}_EER'.format(db_name), eer, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds, val, val_std, far, frr, eer = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor, far.mean(), frr.mean(), eer

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        accuracy_record = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            for imgs, labels in tqdm(iter(self.loader)):
                
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                
                #####################################
                if torch.isnan(loss):
                    
                    sum_emb = torch.sum(embeddings)
                    sum_theta = torch.sum(thetas)
                    self.logger_trainloss.info(
                        'Sum of Embedding NAN Tensor: {}'.format(sum_emb))                    
                    self.logger_trainloss.info(
                        'Sum of Thetas Tensor: {}'.format(sum_theta))
                    raise ValueError("Loss is NAN!!!!!!!!!!!!!!!")
                ###################################################


                if self.step % self.board_loss_every == 0 and self.step != 0:
                    print('GPU IDS: {}'.format(self.gpu_ids))
                    print('Training Model: {}'.format(self.net_type))                    
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('GPU IDS: {}'.format(self.gpu_ids))
                    print('Training Model: {}'.format(self.net_type))
                    accuracy_agedb30, best_threshold_agedb30, roc_curve_tensor_agedb30, far_agedb30, frr_agedb30, eer_agedb30 = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy_agedb30, best_threshold_agedb30, roc_curve_tensor_agedb30, far_agedb30, frr_agedb30, eer_agedb30)
                    accuracy_lfw, best_threshold_lfw, roc_curve_tensor_lfw, far_lfw, frr_lfw, eer_lfw = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy_lfw, best_threshold_lfw, roc_curve_tensor_lfw, far_lfw, frr_lfw, eer_lfw)
                    accuracy_cfp, best_threshold_cfp, roc_curve_tensor_cfp, far_cfp, frr_cfp, eer_cfp = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy_cfp, best_threshold_cfp, roc_curve_tensor_cfp, far_cfp, frr_cfp, eer_cfp)
                    self.model.train()
                    accuracy = (accuracy_agedb30 + accuracy_cfp + accuracy_lfw) / 3
                    if accuracy > accuracy_record:
                        accuracy_record = accuracy
                        self.save_state(conf, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, extra='epoch_{}'.format(e))
