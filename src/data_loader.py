import numpy as np
import cv2
import os
import random
import pandas as pd
import sys

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False, num_classes=10):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        #num_classes: total number of classes into which the crowd count is divided (default: 10 as used in the paper)
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
        self.min_gt_count = sys.maxint
        self.max_gt_count = 0
        self.num_classes = num_classes
        self.count_class_hist = np.zeros(self.num_classes)        
        if self.pre_load:
            self.preload_data() #load input images and grount truth into memory                
            self.assign_gt_class_labels() #assign ground truth crowd group/class labels to each image
            
        else:
            self.get_stats_in_dataset() #get min - max crowd count present in the dataset. used later for assigning crowd group/class
            
    
    def get_classifier_weights(self):
        #since the dataset is imbalanced, classifier weights are used to ensure balance.
        #this function returns weights for each class based on the number of samples available for each class
        wts = self.count_class_hist
        wts = 1-wts/(sum(wts));
        wts = wts/sum(wts);
        return wts
        
    def preload_data(self):
        print 'Pre-loading the data. This may take a while...'
        idx = 0
        for fname in self.data_files:            
            img, den, gt_count = self.read_image_and_gt(fname)
            self.min_gt_count = min(self.min_gt_count, gt_count)
            self.max_gt_count = max(self.max_gt_count, gt_count)
            
            blob = {}
            blob['data']=img
            blob['gt_density']=den
            blob['fname'] = fname                                
            blob['gt_count'] = gt_count
            
            self.blob_list[idx] = blob
            idx = idx+1
            if idx % 100 == 0:                               
                print 'Loaded ', idx , '/' , self.num_samples
        print 'Completed laoding ' ,idx, 'files'
        
        
    def assign_gt_class_labels(self):        
        for i in range(0,self.num_samples):
            gt_class_label = np.zeros(self.num_classes, dtype=np.int)
            bin_val = (self.max_gt_count - self.min_gt_count)/float(self.num_classes)
            class_idx = np.round(self.blob_list[i]['gt_count']/bin_val)
            class_idx = int(min(class_idx,self.num_classes-1))
            gt_class_label[class_idx]=1
            self.blob_list[i]['gt_class_label'] = gt_class_label.reshape(1,gt_class_label.shape[0])
            self.count_class_hist[class_idx] += 1
            
                    
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)        
            else:
                random.shuffle(self.data_files)
                
        files = self.data_files
        id_list = self.id_list
       
        
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:
                                    
                fname = files[idx]
                img, den, gt_count = self.read_image_and_gt(fname)
                gt_class_label = np.zeros(self.num_classes,dtype=np.int)
                bin_val = (self.max_gt_count - self.min_gt_count)/float(self.num_classes)
                class_idx = np.round(gt_count/bin_val)
                class_idx = int(min(class_idx,self.num_classes-1) )             
                gt_class_label[class_idx] = 1
                
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                blob['gt_count'] = gt_count
                blob['gt_class_label'] = gt_class_label.reshape(1,gt_class_label.shape[0])
                
                
                
            yield blob
    
    def get_stats_in_dataset(self):
        
        min_count = sys.maxint
        max_count = 0
        gt_count_array = np.zeros(self.num_samples)
        i = 0
        for fname in self.data_files:
            den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
            den  = den.astype(np.float32, copy=False)
            gt_count = np.sum(den)
            min_count = min(min_count, gt_count)
            max_count = max(max_count, gt_count)
            gt_count_array[i] = gt_count
            i+=1
        
        self.min_gt_count = min_count
        self.max_gt_count = max_count        
        bin_val = (self.max_gt_count - self.min_gt_count)/float(self.num_classes)
        class_idx_array = np.round(gt_count_array/bin_val)
        
        for class_idx in class_idx_array:
            class_idx = int(min(class_idx, self.num_classes-1))
            self.count_class_hist[class_idx]+=1
            
        
    def get_num_samples(self):
        return self.num_samples
                
    def read_image_and_gt(self,fname):
        img = cv2.imread(os.path.join(self.data_path,fname),0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht/4)*4
        wd_1 = (wd/4)*4
        img = cv2.resize(img,(wd_1,ht_1))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))
        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
        den  = den.astype(np.float32, copy=False)
        if self.gt_downsample:
            wd_1 = wd_1/4
            ht_1 = ht_1/4
            den = cv2.resize(den,(wd_1,ht_1))                
            den = den * ((wd*ht)/(wd_1*ht_1))
        else:
            den = cv2.resize(den,(wd_1,ht_1))
            den = den * ((wd*ht)/(wd_1*ht_1))
            
        den = den.reshape((1,1,den.shape[0],den.shape[1]))  
        gt_count = np.sum(den)
        
        return img, den, gt_count
        
            
        