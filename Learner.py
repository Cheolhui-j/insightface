from data.data_pipe import de_preprocess, get_train_loader, get_val_data
#from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from model.model_ir import IRNet
from model.model_mobilefacenet import MobileFaceNet
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
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras,\
    setup_logger, make_dir, print_network
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


class face_learner(object):
    def __init__(self, conf, inference=False):
        make_dir(conf.work_path)        
        make_dir(conf.model_path)
        make_dir(conf.log_path)
        make_dir(conf.save_path)

        setup_logger('configs', conf.work_path, 'configs', level=logging.INFO, screen=True)
        self.logger_configs = logging.getLogger('configs')
        
        setup_logger('trainloss', conf.work_path, 'trainloss', level=logging.INFO, screen=True)
        self.logger_trainloss = logging.getLogger('trainloss')
        
        setup_logger('valid', conf.work_path, 'valid', level=logging.INFO, screen=True)
        self.logger_valid = logging.getLogger('valid')

        if conf.gpu_ids:
            assert torch.cuda.is_available(), 'GPU is not avalialble!'
            #torch.cuda.set_device(conf.gpu_ids[0])
            torch.backends.cudnn.benckmark = True 
            conf.device = torch.device('cuda')
        else:
            conf.device = torch.device('cpu')
        
        self.gpu_ids = conf.gpu_ids
   
        for k, v in conf.items():
            log = '\t{} : {}'.format(k, v)
            self.logger_configs.info(log)

        self.model = None
        self.net_type = '{}_{}'.format(conf.net_mode, conf.net_depth)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)            
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'ir' or conf.net_mode =='ir_se':
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

        #self.model = torch.nn.DataParallel(self.model)

        # if conf.gpu_ids:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=conf.gpu_ids)

        sturct_log1, sturct_log2 = print_network(self.model)
        self.logger_configs.info('Model Architecture')
        self.logger_configs.info(sturct_log1) 
        self.logger_configs.info(sturct_log2)        

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            elif conf.use_ADAM:
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
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(conf.emore_folder)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        
        network = None
        if isinstance(self.model, torch.nn.DataParallel):
            network = self.model.module
        else: network = self.model
        assert network is not None

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        ###############################################
        save_name = os.path.join(
            save_path, 'model_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
        torch.save(state_dict, save_name)
        if not model_only:
            head_module = None
            if isinstance(self.head, torch.nn.DataParallel):
                head_module = self.head.module
            else: head_module = self.head
            assert head_module is not None
            head_state_dict = head_module.state_dict()
            for key, param in head_state_dict.items():
                head_state_dict[key] = param.cpu()
            save_name = os.path.join(
                save_path, 'head_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
            torch.save(head_state_dict, save_name)

            optimizer_module = None
            if isinstance(self.optimizer, torch.nn.DataParallel):
                optimizer_module = self.optimizer.module
            else: optimizer_module = self.head
            assert optimizer_module is not None
            optimizer_state_dict = optimizer_module.state_dict()
            for key, param in head_state_dict.items():
                head_state_dict[key] = param.cpu()
            save_name = os.path.join(
                save_path, 'optimizer_{}_accuracy:{:.4f}_step:{}_{}.pth'.format(self.net_type, accuracy, self.step, extra))
            torch.save(optimizer_state_dict, save_name)

            
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
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
        log = '[Training Step : {}] [DB: {}] Accuracy: {:.4f} Best Threshold: {:.4f} FAR: {:.4f} FRR: {:.4f} EER: {:.4f}'.format(
            self.step, db_name, accuracy, best_threshold, far, frr, eer)
        self.logger_valid.info(log)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
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
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
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
                    log = '[Epoch:{}|{}] [Step:{}] Training Loss : {:.4f}\n'.format(e, epochs, self.step, loss_board)
                    self.logger_trainloss.info(log)                    
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('GPU IDS: {}'.format(self.gpu_ids))
                    print('Training Model: {}'.format(self.net_type))
                    log = '\n\n[Epoch:{}|{}]'.format(e, epochs)
                    self.logger_valid.info(log)
                    accuracy, best_threshold, roc_curve_tensor, far, frr, eer = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor, far, frr,eer)
                    accuracy, best_threshold, roc_curve_tensor, far, frr, eer = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, far, frr, eer)
                    accuracy, best_threshold, roc_curve_tensor, far, frr, eer = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor, far, frr, eer)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum
