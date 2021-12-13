# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:05:01 2021

@author: Ibrahim Khalilullah
"""

import math
import os.path as osp
import random
from collections import OrderedDict
import cv2
#import json
import numpy as np
#import torch
import copy


from utils import gaussian_radius, draw_umich_gaussian, xyxy2xywh


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms
        self.mosaic_border = [-img_size[1] // 2, -img_size[0] // 2]
        
      
    def load_image(self, img_files, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r_w = self.width / w0  # resize image to img_size
        r_h = self.height / h0
        if r_w != 1 or r_h != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r_w < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r_w), int(h0 * r_h)), interpolation=interp)
            
        
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

   

    def load_mosaic_ori(self, img_files, label_files, index_ori):
        
        # loads images in a mosaic

        labels4 = []
        x_s = self.width
        y_s = self.height
        
      
        
        yc, xc = self.height, self.width  # mosaic center x, y
        indices = [index_ori] + [random.randint(0, len(label_files) - 1) for _ in range(3)]  # 3 additional image indices
        
       
        
        for i, index in enumerate(indices):
            
            # Load image
            img, _, (h, w) = self.load_image(img_files, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((y_s * 2, x_s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, x_s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(y_s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, x_s * 2), min(y_s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            label_path = label_files[index]
            x = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 2] = w * (x[:, 2] - x[:, 4] / 2) + padw
                labels[:, 3] = h * (x[:, 3] - x[:, 5] / 2) + padh
                labels[:, 4] = w * (x[:, 2] + x[:, 4] / 2) + padw
                labels[:, 5] = h * (x[:, 3] + x[:, 5] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            
        img4, labels4 = random_perspective(img4, labels4, border=self.mosaic_border)  # border to remove

        return img4, labels4

    def get_data(self, img_files, label_files, index):
        
        
        img, labels = self.load_mosaic_ori(img_files, label_files, index)
        
                
        #### MixUp
        if random.random() < 0.0:
            img2, labels2 = self.load_mosaic_ori(img_files, label_files, random.randint(0, len(label_files) - 1))
            r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
            labels = np.concatenate((labels, labels2), 0)
                
        augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)
        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= self.width
            labels[:, 3] /= self.height
            labels[:, 4] /= self.width
            labels[:, 5] /= self.height
            
        if random.random() > 0.5:
            img = np.fliplr(img)
            if nL > 0:
                labels[:, 2] = 1 - labels[:, 2]
        
        
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
       
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_files[index]

    def __len__(self):
        return self.nF  # number of batches



def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed



def random_perspective(img, targets=(), degrees=5, translate=0.1, scale=(0.5, 1.2), shear=2, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    
    
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    ##### Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    ##### Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    ##### Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    #### Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))


    # Transform label coordinates
    n = len(targets)
    if n:
        ##### warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        ##### create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        
        ##### filter candidates
        i = box_candidates(box1=targets[:, 2:6].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 2:6] = xy[i]
        targets = targets[targets[:, 2] < width]
        targets = targets[targets[:, 4] > 0]
        targets = targets[targets[:, 3] < height]
        targets = targets[targets[:, 5] > 0]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  #### box1(4,n), box2(4,n)
    
    
    ##### Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


class JointDataset(LoadImagesAndLabels):  # for training

    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        
        self.opt = opt        
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            ##print(ds, path)   #######  lacrosse   dataset/lacrosse/lacrosse.train
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                #print(self.img_files[ds])
                self.img_files[ds] = [osp.join(root, x.strip()).replace("\\","/") for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]
            
       
        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                ##print("lb",lb)
                if len(lb) < 1:
                    continue
                
                #print("lb.shape", lb.shape)                
                if len(lb.shape) < 2:
                    
                    img_max = lb[1]
                else:
                    #print("lb[:, 1]", lb[:, 1])
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        
        for i, (k, v) in enumerate(self.tid_num.items()):  ###  for multiple datasets (Mixed) e.g., lacrosse, car 
            #print("k, v", k, v)  ##k, v lacrosse 14.0
            self.tid_start_index[k] = last_index
            last_index += v
            
        #print(last_index)   ####### 14.0
        

        self.nID = int(last_index + 1)  ###############  ## total number of object in a dataset
        self.nds = [len(x) for x in self.img_files.values()]  #### total number of frames
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]  #### classes
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        self.mosaic_border = [-img_size[1] // 2, -img_size[0] // 2]
        
        
       
        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        #img_path = self.img_files[ds][files_index - start_index]
        #label_path = self.label_files[ds][files_index - start_index]
        index = files_index - start_index       
        
        
        imgs, labels, img_path = self.get_data(self.img_files[ds], self.label_files[ds], index)
        
                
        #imgs, labels, img_path = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:                
                labels[i, 1] += self.tid_start_index[ds]
                

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        
        
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_umich_gaussian
        #print("number objects..", num_objs)
        #print('max_object K', self.max_objs)
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))                
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy

        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        
        return ret
