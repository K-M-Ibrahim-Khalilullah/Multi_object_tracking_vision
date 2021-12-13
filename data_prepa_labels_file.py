# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:15:16 2021

@author: Ibrahim Khalilullah
"""
import numpy as np
import os
import cv2
import shutil

seq_dir = 'dataset/lacrosse/train'
seqs = [s for s in os.listdir(seq_dir)]
print(seqs)

tid_last = -1
tid_curr = 0

for seq in seqs:  
    data_info = os.path.join(seq_dir, seq).replace("\\","//")
    data_gt = os.path.join(seq_dir, seq, seq+'.txt').replace("\\","//")
    label_root = os.path.join(seq_dir, seq, 'labels_with_ids').replace("\\","//")
    if not os.path.exists(label_root):
        os.makedirs(label_root)
        
    
    ## this for missing image seq 
    image_root = os.path.join(seq_dir, seq, 'images2').replace("\\","//")
    image_root = os.path.join(seq_dir, seq, 'images').replace("\\","//")
    
    
    #############   data info  
    video_info = open(os.path.join(data_info, 'videoinfo.ini')).read()
    frame_width = int(video_info[video_info.find('imWidth=') + 8:video_info.find('\nimHeight')])
    frame_height = int(video_info[video_info.find('imHeight=') + 9:video_info.find('\nimExt')])
    
    print(frame_width, frame_height)
    
    gt = np.loadtxt(data_gt, dtype=np.float64, delimiter=',')
    
    print(gt)
    
    ######### sort the array best on id
    gt_sort = gt[np.argsort(gt[:, 1])]
    
    #np.savetxt("check_sort_data.txt", gt_sort, delimiter=',', newline = '\n')
    # for row in gt_sort:
    #     print(row[0], row[1], row[2], row[3], row[4], row[5])
    #     #break
    
    

    for fid, tid, x, y, w, h, _, _, _, _ in gt_sort:
        #print(fid, tid, x, y, w, h)
        
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:   
            #print(seq, tid, tid_last)
            #break
            tid_curr += 1
            tid_last = tid
        x += w / 2  #### center  x
        y += h / 2  #####  center y
        label_fpath = os.path.join(label_root, '{:06d}.txt'.format(fid)).replace("\\","/")
        
        '''
        ## this for missing image seq , need to change bcz it take so long time
        image_path = os.path.join(image_root, '{:06d}.png'.format(fid)).replace("\\","/")
        save_path = os.path.join(save_root, '{:06d}.png'.format(fid)).replace("\\","/")
        
        ############## better to use direct copy e.g., shutil copy
        if not os.path.isfile(save_path):
            #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            #cv2.imwrite(save_path, image)
            shutil.copy(image_path, save_root)
        
        '''
        
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / frame_width, y / frame_height, w / frame_width, h / frame_height)
        
        #print("xxxx", x, y)
        #label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            #tid_curr, x, y, w, h)
        
        with open(label_fpath, 'a') as f:
            f.write(label_str)
    f.close()