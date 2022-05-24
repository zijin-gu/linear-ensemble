import numpy as np
import torch
import os 
import pickle 
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import random

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

nsddata_dir = '/home/zg243/nsd/LE/data/nsddata/'
class NSDdataset(Dataset): 
    def __init__(self, mode='train', subject=1, roi='OFA', train_size=None):
        responses = np.load(nsddata_dir + f'S{subject}_{mode}_responses.npy', allow_pickle=True).tolist()
        responses = responses[roi]

        images = np.load(nsddata_dir + f'S{subject}_{mode}_images.npy', mmap_mode='r')
        images = np.moveaxis(images, 1, -1)# (n, 3, 227, 227) -> (n, 227, 227, 3)
        if train_size:
            random.seed(subject)
            train_indices = random.sample(range(len(images)), train_size)
            self.images = images[train_indices]
            self.responses = responses[train_indices]
        else:
            self.images = images
            self.responses = responses
        self.n_neurons =  self.responses.shape[1]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.images[index]).astype(np.float32)
        y = np.asarray(self.responses[index]).astype(np.float32)
        X = preprocess(X)
        return X, y 
    
# class NeuroGendataset(Dataset): 
#     def __init__(self, mode='train', subject=1, roi='OFA', train_size=400):
        
#         if mode == 'train':
#             train_indices = random.sample()
#             images = np.load('./data/ses2/ng%03d_images.npy'%subject)
#             responses = np.load('./data/ses2/ng%03d_responses.npy'%subject, allow_pickle=True).tolist()[roi]
#         elif mode == 'test':
#             images = np.load('./data/ses1/ng%03d_images.npy'%subject)
#             responses = np.load('./data/ses1/ng%03d_responses.npy'%subject, allow_pickle=True).tolist()[roi]
#         self.responses = responses
#         self.images = np.moveaxis(images, 1, -1) # (n, 3, 227, 227) -> (n, 227, 227, 3)
#         self.n_neurons =  self.responses.shape[1]
        
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.images) 
  
#     def __getitem__(self, index):
#         'Generates one sample of data'
#         X = np.asarray(self.images[index]).astype(np.float32)
#         y = np.asarray(self.responses[index]).astype(np.float32)
#         X = preprocess(X)
#         return X, y 
