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

torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

BERT_MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'
# BERT_MODEL = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, num_labels = 3)

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

class BERT(nn.Module):
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

# Training Function
def train(model,
          optimizer,
          train_loader,
          valid_loader,
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
        for (text, labels), _ in train_loader:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
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
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)  
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
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text)
                _, output = output
                raw_scores.extend(F.softmax(output, dim = 1).tolist())
                y_pred.extend(torch.argmax(output, 1).tolist())
    return y_pred, raw_scores

def cross_validate(data_path, file_path):
    for cv in range(1,6):
        print('\n##############################################################')
        print('##########################CV: %d###############################'%(cv))
        print('##############################################################')
        train_data_path = data_path + "/train_fold" + str(cv) + ".csv"
        test_data_path = data_path + "/../test_fold" + str(cv) + ".csv"
        train_data = TabularDataset(path=train_data_path, format='CSV', fields = fields, skip_header = True)
        test_data = TabularDataset(path=test_data_path, format='CSV', fields = fields, skip_header = True)
        train_data1, val_data1 = train_data.split(split_ratio= [0.80, 0.20])
        train_iter = BucketIterator(train_data1, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
                                    device=device, train=True, sort=True, sort_within_batch=True)
        valid_iter = BucketIterator(val_data1, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
                                    device=device, train=True, sort=True, sort_within_batch=True)
        test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
        # train MODEL
        model = BERT().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        train(model=model, optimizer=optimizer, train_loader = train_iter, valid_loader = valid_iter,file_path=file_path + "/model", num_epochs = 10, eval_per_epoch = 10)
        load_checkpoint(file_path + '/model/model.pt', model)
        out_loc = file_path + "/fold" + str(cv) + "_predictions.csv"
        y_pred, _ = evaluate(model, test_iter)
        test = pd.read_csv(test_data_path)
        test["pred"] = y_pred
        test.to_csv(out_loc, index = 0)

data_path = "/path/to/stage2/folder"
cross_validate(data_path,data_path)