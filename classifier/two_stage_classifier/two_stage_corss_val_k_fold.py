import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries
from torchtext.data import Field, Dataset, TabularDataset, BucketIterator, Iterator
# Models
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, AutoModel
# Training
import torch.optim as optim
# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
import json
import os
import csv
import warnings
import numpy as np

torch.manual_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

BERT_MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'
# BERT_MODEL = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, num_labels = 4)

MAX_SEQ_LEN = 100
BATCH_SIZE = 4

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

PRINT = True

if not PRINT:
    warnings.filterwarnings("ignore")

# Fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('label', label_field)]

class BERT_stage1(nn.Module):
    def __init__(self):
        # super(BERT, self).__init__()
        super().__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    def forward(self, text, label= None):
        output = self.encoder(text, labels=label)[:2]
        if not label is None:
            loss, text_fea = output
        else:
            text_fea = output[0]
            loss = None
        return loss, text_fea

class BERT_stage2(nn.Module):
    def __init__(self):
        # super(BERT, self).__init__()
        super().__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 3)
    def forward(self, text, label= None):
        output = self.encoder(text, labels=label)[:2]
        if not label is None:
            loss, text_fea = output
        else:
            text_fea = output[0]
            loss = None
        return loss, text_fea

# # Training
# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    
    if PRINT: print(f'Model saved to ==> {save_path}')

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    if PRINT: print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    if PRINT: print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

destination_folder = "/path/to/destination/folder"
destination_folder_stage1 = destination_folder + "/stage1"
destination_folder_stage2 = destination_folder + "/stage2"
# Training Function
def train(model,
          optimizer,
          train_loader,
          valid_loader,
          clf_stage,
          file_path,
          num_epochs = 30,
          eval_per_epoch = 2,
          best_valid_loss = float("Inf")):
    eval_every = len(train_loader) // eval_per_epoch
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    # training loop
    model.train()
    for epoch in range(num_epochs):
        text_np = np.array([[]])
        label_np = np.array([])
        for (text, labels), _ in train_loader:
            labels_numpy = labels.cpu().numpy()
            if clf_stage == 1:
                labels = np.where(np.isin(labels_numpy, np.array([0,1,2])), np.full(labels_numpy.shape, 0), np.full(labels_numpy.shape, 1))
            elif clf_stage == 2:
                text_numpy = text.cpu().numpy()
                labels1 = labels_numpy[np.isin(labels_numpy, np.array([0,1,2]))]
                text1 = text_numpy[np.isin(labels_numpy, np.array([0,1,2]))]
                if len(label_np) < BATCH_SIZE:
                    if label_np.size != 0:
                        x = BATCH_SIZE - len(label_np)
                        label_np = np.concatenate((label_np,labels1[:x]))
                        text_np = np.concatenate((text_np,text1[:x]), axis = 0)
                    else:
                        x = BATCH_SIZE - len(label_np)
                        label_np = labels1
                        text_np = text1
                    if len(label_np) != BATCH_SIZE:
                        continue
                text = torch.from_numpy(text_np)
                labels = label_np
                text_np = text1[x:]
                label_np = labels1[x:]
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.LongTensor)
            text = text.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.to(device)
            output = model(text, labels)
            loss, _ = output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update running values
            running_loss += loss.item()
            global_step += 1
            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                  
                    # validation loop
                    for (text, labels), _ in valid_loader:
                        labels_numpy = labels.cpu().numpy()
                        if clf_stage == 1:
                            labels = np.where(np.isin(labels_numpy, np.array([0,1,2])), np.full(labels_numpy.shape, 0), np.full(labels_numpy.shape, 1))
                        elif clf_stage == 2:
                            text_numpy = text.cpu().numpy()
                            labels1 = labels_numpy[np.isin(labels_numpy, np.array([0,1,2]))]
                            text1 = text_numpy[np.isin(labels_numpy, np.array([0,1,2]))]
                            if len(label_np) < BATCH_SIZE:
                                if label_np.size != 0:
                                    x = BATCH_SIZE - len(label_np)
                                    label_np = np.concatenate((label_np,labels1[:x]))
                                    text_np = np.concatenate((text_np,text1[:x]), axis = 0)
                                else:
                                    x = BATCH_SIZE - len(label_np)
                                    label_np = labels1
                                    text_np = text1
                                if len(label_np) != BATCH_SIZE:
                                    continue
                            text = torch.from_numpy(text_np)
                            labels = label_np
                            text_np = text1[x:]
                            label_np = labels1[x:]
                        labels = torch.from_numpy(labels)
                        labels = labels.type(torch.LongTensor)           
                        text = text.type(torch.LongTensor) 
                        labels = labels.to(device) 
                        text = text.to(device)
                        output = model(text, labels)
                        loss, _ = output
                        valid_running_loss += loss.item()
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
                # print progressi
                if PRINT:
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                                  average_train_loss, average_valid_loss))
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    raw_scores = []
    model.eval()
    with torch.no_grad():
        for (text, labels), _ in test_loader:
                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text)
                _, output = output
                raw_scores.extend(F.softmax(output, dim = 1).tolist())
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    return y_pred,y_true,raw_scores

def evaluate_two_stage(model1, model2,test_loader):
    y_pred1,y_true1,raw_scores1 = evaluate(model1, test_loader)
    y_pred2,y_true2,raw_scores2 = evaluate(model2, test_loader)
    y_pred_final = np.where(np.isin(y_pred1, np.array([1])),np.full(len(y_pred1), 3), y_pred2)
    tmp = classification_report(y_true1, y_pred_final, digits=4, output_dict= True)
    print(json.dumps(tmp))
    print(confusion_matrix(y_true1, y_pred_final))

def cross_validate(data_path):
    data = TabularDataset(path=data_path, format='CSV', fields = fields, skip_header = True)
    d = [None] * 5
    tmp, d[3], d[4] = data.split(split_ratio = [0.6,0.2,0.2])
    d[0], d[1], d[2] = tmp.split(split_ratio = [0.33,0.33,0.33])
    for cv in range(5):
        print('\n##############################################################')
        print('##########################CV: %d###############################'%(cv + 1))
        print('##############################################################')
        tmp = []
        for i in range(5):
            if i != cv:
                tmp.extend(list(d[i]))
        train_data = Dataset(tmp, fields)
        val_data = d[cv]
        train_data1, val_data1 = train_data.split(split_ratio= [0.75, 0.25])
        train_iter = BucketIterator(train_data1, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
                                    device=device, train=True, sort=True, sort_within_batch=True)
        valid_iter = BucketIterator(val_data1, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
                                    device=device, train=True, sort=True, sort_within_batch=True)
        test_iter = Iterator(val_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
        # train first stage MODEL
        model_stage1 = BERT_stage1().to(device)
        optimizer_stage1 = optim.Adam(model_stage1.parameters(), lr=1e-6)
        train(model=model_stage1, optimizer=optimizer_stage1, train_loader = train_iter, valid_loader = valid_iter, clf_stage = 1,file_path = destination_folder_stage1, num_epochs = 1, eval_per_epoch = 10)
        del model_stage1
        torch.cuda.empty_cache()
        # train second stage MODEL
        model_stage2 = BERT_stage2().to(device)
        optimizer_stage2 = optim.Adam(model_stage2.parameters(), lr=1e-6)
        train(model=model_stage2, optimizer=optimizer_stage2, train_loader = train_iter, valid_loader = valid_iter, clf_stage = 2, file_path = destination_folder_stage2, num_epochs = 1, eval_per_epoch = 10)
        del model_stage2
        torch.cuda.empty_cache()
        #load the best models from both the stages for evaluation
        model_stage1 = BERT_stage1().to(device)
        load_checkpoint(destination_folder_stage1 + '/model.pt', model_stage1)
        model_stage2 = BERT_stage2().to(device)
        load_checkpoint(destination_folder_stage2 + '/model.pt', model_stage2)
        #two stage evaluation for the entire pipeline
        evaluate_two_stage(model_stage1, model_stage2, test_iter)

data_path = "/path/to/multiclass_training/data"
cross_validate(data_path)