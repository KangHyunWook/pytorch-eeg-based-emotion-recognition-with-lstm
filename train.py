'''
Programmer: Hyunwook Kang
Date: 23-Aug-2022
Description: this program predicts emotions(positive, neutral, negative) from eeg signals in human brains.
'''
from torch import optim

import pandas as pd
import argparse
import os
import pickle
import shutil
import numpy as np
from random import random
from torch.utils.data import DataLoader, Dataset

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

#code adapted from https://github.com/declare-lab/MISA/blob/master/src/utils/convert.py
def to_gpu(x, on_cpu=False, gpu_id=None):

    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.output_size = 3
        
        rnn = nn.LSTM 

        self.eeg_rnn = rnn(2548, 256, bidirectional=False)
        self.eeg_rnn.add_module('eeg_rnn_dropout', nn.Dropout(0.2))
        self.eeg_rnn.add_module('eeg_rnn', rnn(256, 256,bidirectional=False))
        self.eeg_rnn.add_module('eeg_rnn_dropout2', nn.Dropout(0.2))

        self.project = nn.Sequential()

        self.project.add_module('project', 
                nn.Linear(in_features=256, out_features=self.output_size))
        self.project.add_module('project_activation', 
                                                nn.ReLU())
        self.project.add_module('project_softmax', nn.Softmax())
                                    
    def forward(self, eeg_samples):
        # batch_size = lengths.size(0)
        batch_size=eeg_samples.size(1)
        
        h1, (h2, _) = self.eeg_rnn(eeg_samples)
        
        eeg_feats = self.project(h1.view(batch_size,-1))
        
        return eeg_feats

def my_eval(mode=None, to_print=False):
    assert(mode is not None)
    
    model.eval()

    y_true, y_pred = [], []
    eval_loss, eval_loss_diff = [], []
  
    if mode == "dev":
        target_data = dev_batch_data
        target_labels =dev_batch_labels
    elif mode == "test":
        target_data, target_labels= test_batch_data, test_batch_labels

        if to_print:
            model.load_state_dict(torch.load(
                f'checkpoints/model_saved.pth'))
            
    with torch.no_grad():
        correct=0
        for i in range(len(target_data)):
            model.zero_grad()
            a=torch.from_numpy(target_data[i]).float() 
            y=torch.from_numpy(target_labels[i]).float()
            
            a=a.reshape((1, bs, -1))
           
            a = to_gpu(a)
           
            labels=torch.zeros((bs,3))
            for i in range(labels.size(0)):
                labels[i,y[i].long()]=1

            y = to_gpu(labels)

            y_tilde = model(a)

            criterion=nn.CrossEntropyLoss(reduction='mean')

            cls_loss = criterion(y_tilde, y)
            loss = cls_loss
            
            eval_loss.append(loss.item())
            

            pred=y_tilde.detach().cpu().numpy().astype(np.float)
            
            true=y.detach().cpu().numpy()
            for j in range(pred.shape[0]):
                if(pred[j].argmax()==true[j].argmax()):
                    correct+=1

    eval_loss = np.mean(eval_loss)
    
    accuracy =correct/(dev_batch_data.shape[0]*dev_batch_data.shape[1])
    print('accuracy: ', accuracy)
    return eval_loss, accuracy


def get_batch_data(data,labels,n):
    batch_data=[]
    batch_labels=[]
    for i in range(0, len(data)-n, bs):
        if(i+bs<len(data)):
            batch_data.append(data[i:i+bs].astype(np.float))
            batch_labels.append(labels[i:i+bs].astype(np.float))
        else:
            batch_data.append(data[i:].astype(np.float))
            batch_labels.append(labels[i:].astype(np.float))
            
    return np.array(batch_data), np.array(batch_labels)
    
if __name__ == '__main__':
    
    if(os.path.exists('checkpoints')): shutil.rmtree('checkpoints')
    # Setting random seed
    # random_seed = 336   
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    
    model = Model()

    df_eeg=pd.read_csv('emotions.csv')

    if torch.cuda.is_available():
        model.cuda()
    
    criterion = nn.CrossEntropyLoss(reduction="mean")
    
    best_valid_loss = float('inf')

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001) 

    labels_map={'NEGATIVE':0, 'NEUTRAL':1, 'POSITIVE':2};
    
    data=df_eeg.iloc[:,:-1].values
    labels=df_eeg.iloc[:,-1].values
    temp_labels=[]
    
    for label in labels:
        temp_labels.append(labels_map[label])
    # labels=labels.reshape(labels.shape+(1,))
    batch_data=[]
    batch_labels=[]
    
    labels=np.array(temp_labels)
   
    n_train_data=int(len(data)*0.6)
    n_test_data=int(len(data)*0.2)
    n_dev_data=len(data)-n_test_data-n_train_data
    
    train_data=data[:n_train_data]
    train_labels=labels[:n_train_data]
    
    dev_data=data[n_train_data:n_train_data+n_dev_data]
    dev_labels=labels[n_train_data:n_train_data+n_dev_data]
    
    test_data=data[n_train_data+n_dev_data:]
    test_labels=labels[n_train_data+n_dev_data:]
    
    print(train_data.shape)
    print(dev_data.shape)
    print(test_data.shape)
    
    bs=34

    train_batch_data, train_batch_labels = get_batch_data(train_data,train_labels, 21)   
    dev_batch_data, dev_batch_labels = get_batch_data(dev_data,dev_labels,19)
    test_batch_data, test_batch_labels = get_batch_data(test_data,test_labels,18)
    
    #train
    for e in range(30):
        model.train()

        train_loss_cls=[]
        
        for i in range(len(train_batch_data)):
            model.zero_grad()
            a=torch.from_numpy(train_batch_data[i]).float()
            y=torch.from_numpy(train_batch_labels[i]).float()

            #one-hot encoding
            labels=torch.zeros((bs,3))
            for i in range(labels.size(0)):
                labels[i,y[i].long()]=1
      
            a=a.reshape((1, bs, -1))
         
            a = to_gpu(a)

            y = to_gpu(labels)
              
            y_tilde = model(a)
            
            cls_loss = criterion(y_tilde, y)
            
            cls_loss.backward()
            
            optimizer.step()

            train_loss_cls.append(cls_loss.item())
        
        print(f"Training loss: {round(np.mean(train_loss_cls), 4)}")
        
        valid_loss, valid_acc = my_eval(mode="dev")
                
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/model_saved.pth')
            torch.save(optimizer.state_dict(), f'checkpoints/optim_saved.pth')
        
           
    my_eval(mode="test", to_print=True)
