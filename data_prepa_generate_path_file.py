# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 21:35:09 2021

@author: Ibrahim Khalilullah
"""

import numpy as np
import os
from tqdm import tqdm

def gen_train_val_pathfile(data_root, image_dir_path, out_root, file_name='visdrone.train'):
    
    ##################  this function is for image sequences from multiple videos  ###
    
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('Directory is not correct')
        return

    out_f_path = out_root + '/' + file_name
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + image_dir_path
        seqs = [x for x in os.listdir(root)]
        print(seqs)
        seqs.sort()
        # seqs = sorted(seqs, key=lambda x: int(x.split('_')[-1]))
        for seq in tqdm(seqs):
            img_dir = root + '/' + seq + '/images'
            img_list = [x for x in os.listdir(img_dir)]
            img_list.sort()
            for img in img_list:
                if img.endswith('.png'):
                    img_path = img_dir + '/' + img
                    if os.path.isfile(img_path):
                        ####### check the image path first ##########
                        #item = img_path.replace(data_root + '/', '')
                        # print(item)
                        #f.write(item + '\n')
                        #print(img_path[2:])  ############  check the path 
                        ######break
                        print('saving path: ', img_path)
                        f.write(img_path + '\n')
                        cnt += 1

    print('Total {:d} images for training'.format(cnt))



if __name__=='__main__':
    
    data_root = 'dataset/lacrosse'
    #val_path_file = 'dataset/visdrone/val'
    image_dir_path_train = '/train'
    #image_dir_path_val = '/val/images'
    
    train_path_file = 'dataset/lacrosse'
    
    
    gen_train_val_pathfile(data_root, image_dir_path_train, train_path_file, file_name = 'lacrosse.train')