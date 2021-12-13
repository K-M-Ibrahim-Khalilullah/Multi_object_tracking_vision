# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:05:24 2021

@author: Ibrahim Khalilullah
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from dotmap import DotMap
from torchvision.transforms import transforms as T
from dataset_loader_train import JointDataset

import os
import torch
import torch.nn as nn
from logger import Logger
from pathlib import Path
import math
from copy import deepcopy
import time
from progress.bar import Bar
import torch.nn.functional as F
from utils import AverageMeter


def update_dataset_info_and_set_heads(parameters_settings, dataset):
      
             
    input_h, input_w = dataset.default_resolution
    parameters_settings.mean, parameters_settings.std = dataset.mean, dataset.std
    parameters_settings.num_classes = dataset.num_classes

    # input_h(w): parameters_settings.input_h overrides parameters_settings.input_res overrides dataset default
    input_h = parameters_settings.input_res if parameters_settings.input_res > 0 else input_h
    input_w = parameters_settings.input_res if parameters_settings.input_res > 0 else input_w
    parameters_settings.input_h = parameters_settings.input_h if parameters_settings.input_h > 0 else input_h
    parameters_settings.input_w = parameters_settings.input_w if parameters_settings.input_w > 0 else input_w
    parameters_settings.output_h = parameters_settings.input_h // parameters_settings.down_ratio
    parameters_settings.output_w = parameters_settings.input_w // parameters_settings.down_ratio
    parameters_settings.input_res = max(parameters_settings.input_h, parameters_settings.input_w)
    parameters_settings.output_res = max(parameters_settings.output_h, parameters_settings.output_w)

   
    parameters_settings.heads = {'hm': parameters_settings.num_classes,
                 'wh': 2 if not parameters_settings.ltrb else 4,
                 'id': parameters_settings.reid_dim}
    if parameters_settings.reg_offset:
      parameters_settings.heads.update({'reg': 2})
    parameters_settings.nID = dataset.nID
    #parameters_settings.img_size = (1088, 608)
    #parameters_settings.img_size = (864, 480)
    #parameters_settings.img_size = (576, 320)
    parameters_settings.img_size = (1920, 1056)
    #parameters_settings.img_size = (640, 352)   
      
    print('heads', parameters_settings.heads)
    print("inside update_dataset...input_h w input_res res", parameters_settings.input_h, parameters_settings.input_w, parameters_settings.input_res, parameters_settings.output_res)
    print("inside update_dataset... output_h, output_w:", parameters_settings.output_h, parameters_settings.output_w)
    
    '''
    heads {'hm': 1, 'wh': 4, 'id': DotMap(), 'reg': 2}
    inside update_dataset...input_h w input_res res 1088 608 1088 272
    inside update_dataset... output_h, output_w: 272 152
    '''
    return parameters_settings

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
      start_epoch = 0
      checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
      print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
      state_dict_ = checkpoint['state_dict']
      state_dict = {}
      
      # convert data_parallal to model
      for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
          state_dict[k[7:]] = state_dict_[k]
        else:
          state_dict[k] = state_dict_[k]
      model_state_dict = model.state_dict()
    
      # check loaded parameters and created model parameters
      msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
      for k in state_dict:
        if k in model_state_dict:
          if state_dict[k].shape != model_state_dict[k].shape:
            print('Skip loading parameter {}, required shape{}, '\
                  'loaded shape{}. {}'.format(
              k, model_state_dict[k].shape, state_dict[k].shape, msg))
            state_dict[k] = model_state_dict[k]
        else:
          print('Drop parameter {}.'.format(k) + msg)
      for k in model_state_dict:
        if not (k in state_dict):
          print('No param {}.'.format(k) + msg)
          state_dict[k] = model_state_dict[k]
      model.load_state_dict(state_dict, strict=False)
    
      # resume optimizer parameters
      if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
          optimizer.load_state_dict(checkpoint['optimizer'])
          start_epoch = checkpoint['epoch']
          start_lr = lr
          for step in lr_step:
            if start_epoch >= step:
              start_lr *= 0.1
          for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
          print('Resumed optimizer with start lr', start_lr)
        else:
          print('No optimizer parameters in checkpoint.')
          
      if optimizer is not None:
        return model, optimizer, start_epoch
      else:
        return model

def save_model(path, epoch, model, optimizer=None):
      if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
      else:
        state_dict = model.state_dict()
      data = {'epoch': epoch,
              'state_dict': state_dict}
      if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
      torch.save(data, path)
      
###############################################################################
###############################################################################

#### Model class 

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
        
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
        

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
       
def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        #######print("inside parse model: ", m)
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, SPP, Focus, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class Model(nn.Module):
    def __init__(self, config='configs/yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        ###print(config)
        if isinstance(config, dict):
            self.yaml = config  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(config).name
            with open(config) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class PoseYOLOv5s(nn.Module):
    def __init__(self, heads, config_file):
        self.heads = heads
        super(PoseYOLOv5s, self).__init__()
        self.backbone = Model(config_file)
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(64, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

    def forward(self, x):
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def create_model(heads, config_path, pretrained_path):
    config_file = os.path.join(config_path)
    pretrained = os.path.join(pretrained_path)
    model = PoseYOLOv5s(heads, config_file)
    initialize_weights(model, pretrained)
    return model


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def initialize_weights(model, pretrained=''):
    for i, m in enumerate(model.modules()):
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

    for head in model.heads:
        final_layer = model.__getattr__(head)
        for i, m in enumerate(final_layer.modules()):
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[0] == model.heads[head]:
                    if 'hm' in head:
                        nn.init.constant_(m.bias, -2.19)
                    else:
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)

    if os.path.isfile(pretrained):
        ckpt = torch.load(pretrained)  # load checkpoint
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.backbone.state_dict())  # intersect
        model.backbone.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), pretrained))  # report

############################# Multi-object Tracking Trainer ##################
##############################################################################
############# Utility functions for training #################################

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

#### Losses functions

def _neg_loss(pred, gt):
      ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
      '''
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
    
      neg_weights = torch.pow(1 - gt, 4)
    
      loss = 0
    
      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()
    
      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

class FocalLoss(nn.Module):
      '''nn.Module warpper for focal loss'''
      def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
    
      def forward(self, out, target):
        return self.neg_loss(out, target)

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss


class ModelWithLoss(torch.nn.Module):
      def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
  
      def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class BaseTrainer(object):
    
      def __init__(
        self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.optimizer.add_param_group({'params': self.loss.parameters()})
    
      def set_device(self, gpus, device):
          
          self.model_with_loss = self.model_with_loss.to(device)
          for state in self.optimizer.state.values():
              for k, v in state.items():
                if isinstance(v, torch.Tensor):
                  state[k] = v.to(device=device, non_blocking=True)

      def run_epoch(self, phase, epoch, data_loader):
              
            model_with_loss = self.model_with_loss
            ###if phase == 'train':
            model_with_loss.train()
              
            '''
            else:
              
              model_with_loss.eval()
              torch.cuda.empty_cache()
            '''
        
            opt = self.opt
            results = {}
            data_time, batch_time = AverageMeter(), AverageMeter()
            avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
            num_iters = len(data_loader)
            
            bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
            end = time.time()
            for iter_id, batch in enumerate(data_loader):
              if iter_id >= num_iters:
                break
              data_time.update(time.time() - end)
        
              for k in batch:
                if k != 'meta':
                  batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        
              output, loss, loss_stats = model_with_loss(batch)
              loss = loss.mean()
              
              
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
                
              batch_time.update(time.time() - end)
              end = time.time()
        
              Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
              
              for l in avg_loss_stats:
                avg_loss_stats[l].update(
                  loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                
              if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                  
              if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                  print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
              else:
                bar.next()            
                
              del output, loss, loss_stats, batch
            
            bar.finish()
            
            ret = {k: v.avg for k, v in avg_loss_stats.items()}
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            
            return ret, results     
      
    
      def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)


class MotLoss(torch.nn.Module):
    
    def __init__(self, opt):
        
        super(MotLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss() 
        self.crit_wh = self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
                    
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            ########if not opt.mse_loss:
            output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            
            id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
            id_head = id_head[batch['reg_mask'] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][batch['reg_mask'] > 0]

            id_output = self.classifier(id_head).contiguous()
            id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats

class MTtrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MTtrainer, self).__init__(opt, model, optimizer=optimizer)

    
    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    '''
    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
    '''

###############################################################################
  
def main(param, dataset_root, trainset_paths):
    
    
    ########## set and check GPU
    torch.manual_seed(param.seed)
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    my_visible_devs = '0'  # '0, 3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
    param.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("CPU or GPU: ", param.device)
    param.gpus = my_visible_devs    
    logger = Logger(param)
    
    #### step 1: Data preparation for loading into pytorch object
    
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(param, dataset_root, trainset_paths, img_size = (1920, 1056), augment=True, transforms=transforms)
    param = update_dataset_info_and_set_heads(param, dataset) ##### update param based on the dataset
    
    ##############   Step 2: Training #########################################
    ###########################################################################
    #####  Get dataloader in pytorch ##########
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=param.batch_size,
        shuffle=True,
        num_workers=param.num_workers,
        pin_memory=True,
        drop_last=True
    )

    ##### Model creation for detection
    print('Creating model...')
    model = create_model(param.heads, param.train_cfg, param.yolo_model)
    optimizer = torch.optim.Adam(model.parameters(), param.lr)
    start_epoch = 0
    
    ############  Embedding the model with Multi-object Tracking
    trainer = MTtrainer(param, model, optimizer)
    trainer.set_device(param.gpus, param.device)
    
    if param.load_model != '':
        print('loading weight by base model...')
        model, optimizer, start_epoch = load_model(model, param.load_model, trainer.optimizer,\
                                                   param.resume, param.lr, param.lr_step)
    print('Starting training...')
    print('start_epoch:', start_epoch)
    
    for epoch in range(start_epoch + 1, param.num_epochs + 1):
        
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        
        if epoch % param.save_intervals == 0:
            save_model(os.path.join(param.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
     
        logger.write('\n')
        
        if epoch in param.lr_step:
            save_model(os.path.join(param.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)
            lr = param.lr * (0.1 ** (param.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
    save_model(os.path.join(param.save_dir, 'model_last.pth'), epoch, model, optimizer)                
        
    logger.close()



if __name__=='__main__':
    
    param = DotMap()
    
    #######  set and check GPU
    param.seed = 317
    
    
    
    ########### step 1:  parameters for dataset preparation     
    param.data_path = 'dataset/lacrosse/lacrosse.json' 
    param.K = 500   ########  maximum number of output objects
    param.down_ratio = 4  ######  output stride
    param.ltrb = True   ############  regress left, top, right, bottom of bbox
    ###param.mse_loss = False
    
    f = open(param.data_path)
    data_config = json.load(f)
    trainset_paths = data_config['train']    
    dataset_root = data_config['root']
    f.close()
    
    #### Step 2: Training parameters ########    
        
    param.num_epochs = 1
    param.resume = False
    param.lr = 5e-4
    param.lr_step = [10, 20]  #### drop learning rate by 10
    param.fix_res = True  ####  keep the original resolution during validation
    param.reg_offset = True   ####  not regress local offset
    param.head_conv = -1    ### # -1 for init default head_conv, 0 for no conv layer, need to check without it
    param.batch_size = 4
    param.input_h = -1 ### 'input height. -1 for default from dataset.'
    param.input_w = -1 ### 'input width. -1 for default from dataset.'
    param.input_res = -1  #### 'input height and width. -1 for default from dataset'
    param.reid_dim = 128  ### feature dim for reid
    param.num_stacks = 1
    param.off_weight = 1   #### loss weight for keypoint local offsets
    param.hm_weight = 1    ### loss weight for keypoint heatmaps.
    param.wh_weight = 0.1  ### loss weight for bounding box size.
    param.multi_loss = 'uncertainty'    #### multi_task loss: uncertainty
    
    ####### visualization, save folder and path during training
    param.save_intervals = 1
    param.hide_data_time = False   ##### not display time during training
    param.print_iter = 0   ##### disable progress bar and print to screen.
    param.root_dir = 'train_model' 
    param.task = 'mot'  ####### multiobject tracking folder
    param.exp_id = 'lacrosse'  ### Project
    
    param.train_cfg = 'configs/yolov5s.yaml'
    param.yolo_model = 'pretrained/yolo5s.pt'
    param.load_model = 'pretrained/model_last.pth'
    
    param.exp_dir = os.path.join(param.root_dir, param.task).replace("\\","/")
    param.save_dir = os.path.join(param.exp_dir, param.exp_id).replace("\\","/")
    param.debug_dir = os.path.join(param.save_dir, 'debug').replace("\\","/")
    print('The output will be saved to ', param.save_dir)
    
    ##########  for multiprocessing, 0 for without multiprocessing
    param.num_workers = 4 
    
    
    main(param, dataset_root, trainset_paths)