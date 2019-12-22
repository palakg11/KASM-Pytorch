#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gensim
import numpy as np
import pickle as cPickle
import torch.optim as optim
import time
import sys
from collections import defaultdict


# In[2]:


import torch.nn as nn
import torch.nn.functional as F

class LSTMcell(nn.Module):
    
    def __init__(self,input_size, hidden_size, output_size, cell):
        
        super(LSTMcell, self).__init__()
        
        self.hidden_size = hidden_size
        self.cell = cell
        
        """
        LSTM cell basic operations
        """
        self.i2cdasht = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
        if cell == "RKM-LSTM" or "LSTM":
            self.i2ft = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2it = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if cell == "RKM-CIFG":
            self.i2ft = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if cell == "Linear-Kernel-wto" or "Gated-CNN":
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if cell == "Linear-kernel-wto" or "Linear-Kernel" or "Gated-CNN" or "CNN":
            self.sigmai = 0.5
        if cell == "Linear-kernel-wto" or "Linear-Kernel":
            self.sigmaf = 0.5


    def forward(self, input, hidden_state, cell_state):
        
        """
        input dimension = (batch size X 300); where 300 is dimension used for word embedding
        
        hidden state dimension = (batch size X 300); where 300 is hidden state dimension as mentioned in the paper
        
        """
        combined = torch.cat((input, hidden_state), axis = 1)
        
        if self.cell == "LSTM" or "RKM-LSTM" or "RKM-CIFG":
            forget_gate = torch.sigmoid(self.i2ft(combined))
        if self.cell == "LSTM" or "RKM-LSTM":
            i_t = torch.sigmoid(self.i2it(combined))
        c_dash = self.i2cdasht(combined)
        
        if self.cell == "LSTM":
            cell_state = forget_gate*cell_state + i_t*torch.tanh(c_dash)
        if self.cell == "RKM-LSTM":
            cell_state = forget_gate*cell_state + i_t*c_dash
        if self.cell == "RKM-CIFG":
            cell_state = forget_gate*cell_state + (1 - forget_gate)*c_dash
        if self.cell == "Linear-kernel-wto" or "Linear-kernel":
            cell_state = self.sigmai*c_dash + self.sigmaf*cell_state
        if self.cell == "Gated-CNN" or "CNN":
            cell_state = self.sigmai*c_dash
        
        """
        IMP: Layer normalization [2] to be performed after the computation of the cell state
        """
        if self.cell == "LSTM" or "RKM-LSTM" or "RKM-CIFG" or "Linear-Kernel-wto" or "Gated-CNN":
            output_state = torch.sigmoid(self.i2o(combined))
        if self.cell == "LSTM":
            hidden_state = output_state*torch.tanh(cell_state)
        if self.cell == "RKM-LSTM" or "RKM-CIFG" or "Linear-kernel-wto" or "Gated-CNN":
            hidden_state = output_state*cell_state
        if self.cell == "Linear-Kernel" or "CNN":
            hidden_state = torch.tanh(cell_state)
        
        
        return hidden_state, cell_state


# In[3]:


class LSTMclassifier(nn.Module):
    
    """
    Classification task on LSTM output
    """
    
    def __init__(self,input_size, hidden_size, output_size, glove_weights, cell):
        
        super(LSTMclassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.labels = output_size
        
        """
        Glove embeddings initialization
        """
        self.embedding = nn.Embedding.from_pretrained(glove_weights)
        
        self.lstm = LSTMcell(input_size ,hidden_size, output_size, cell)
        
        """
        Pooling layer: mean pooling across time 
                       pooling layer's input dimension: (batch_size X max_num_of_words X 300)
                       pooling layer's output dimension: (batch_size X 300)
        
        """
        """
        taking intermediate layer size = 100
        """
        self.layer1 = nn.Linear(self.hidden_size, 100, bias = True)
        self.layer2 = nn.Linear(100, self.labels, bias = True)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, max_num_of_words):
        
        input = (self.embedding(input)).float()
        batch_size = input.size()[0]
        hidden_state = torch.zeros(batch_size, self.hidden_size)
        cell_state = torch.zeros(batch_size, self.hidden_size)
        
        """
        output is concatenation of hidden state at all time stamp
        """
        output = torch.zeros((batch_size, max_num_of_words, 300))
        if torch.cuda.is_available():
            output = output.cuda()
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        
        for i in range(max_num_of_words):
            hidden_state, cell_state = self.lstm(input[:,i,:], hidden_state, cell_state)
            output[:,i,:] = hidden_state
        
        pool = nn.AvgPool2d((max_num_of_words,1), stride=1)
        time_avg_output = torch.squeeze(pool(output))
        linear_layer = torch.sigmoid(self.layer1(time_avg_output))
        final_output = torch.sigmoid(self.layer2(linear_layer))
        final_output = self.softmax(final_output)
        
        return final_output


# In[5]:


class Dataset(object):
    
    def load_data(self, dataset):
        
        self.data = dataset
        
        if self.data == 'yahoo':
            self.loadpath = "./data/LEAM_dataset/yahoo.p"
            self.embpath = "./data/LEAM_dataset/yahoo_glove.p"
            self.num_class = 10
            self.class_name = ['Society Culture',
                'Science Mathematics',
                'Health' ,
                'Education Reference' ,
                'Computers Internet' ,
                'Sports' ,
                'Business Finance' ,
                'Entertainment Music' ,
                'Family Relationships' ,
                'Politics Government']
        elif self.data == 'agnews':
            self.loadpath = "./data/LEAM_dataset/ag_news.p"
            self.embpath = "./data/LEAM_dataset/ag_news_glove.p"
            self.num_class = 4
            self.class_name = ['World',
                            'Sports',
                            'Business',
                            'Science']    
        elif self.data == 'dbpedia':
            self.loadpath = "./data/LEAM_dataset/dbpedia.p"
            self.embpath = "./data/LEAM_dataset/dbpedia_glove.p"
            self.num_class = 14
            self.class_name = ['Company',
                'Educational Institution',
                'Artist',
                'Athlete',
                'Office Holder',
                'Mean Of Transportation',
                'Building',
                'Natural Place',
                'Village',
                'Animal',
                'Plant',
                'Album',
                'Film',
                'Written Work',
                ]
        elif self.data == 'yelp_full':
            self.loadpath = "./data/LEAM_dataset/yelp_full.p"
            self.embpath = "./data/LEAM_dataset/yelp_full_glove.p"
            self.num_class = 5
            self.class_name = ['worst',
                            'bad',
                            'middle',
                            'good',
                            'best']

        x = cPickle.load(open(self.loadpath, "rb"), encoding = "latin1")
        self.train, self.val, self.test = x[0], x[1], x[2]
        self.train_lab, self.val_lab, self.test_lab = x[3], x[4], x[5]
        self.wordtoix, self.ixtoword = x[6], x[7]
        del x
        
        print("load data finished:", self.data)


# In[6]:


def eval_model(model, data, label, batch_size):
    total_epoch_loss = 0
    total_epoch_acc = 0
    loss_fn = nn.NLLLoss()
    model.eval()
    steps = 0
    
    with torch.no_grad():
        for iter in range(0, len(data), batch_size):
            text = torch.nn.utils.rnn.pad_sequence(data[iter:min((iter+batch_size), len(data))], batch_first = True)
            target = label[iter:min((iter+batch_size), len(data))].long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text, text.size()[1])
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/min(batch_size, (len(data) - iter))
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            steps += 1
    
    return total_epoch_loss/steps, total_epoch_acc/steps


# In[7]:


def train_model(model, data, label, batch_size, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    if torch.cuda.is_available():
        model.cuda()
        
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    steps = 0
    loss_fn = nn.NLLLoss()
    model.train()
    
    for iter in range(0, len(data), batch_size):
        text = torch.nn.utils.rnn.pad_sequence(data[iter:min((iter+batch_size), len(data))], batch_first = True)
        target = label[iter:min((iter+batch_size), len(data))].long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        optim.zero_grad()
        prediction = model(text, text.size()[1])
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
        acc = 100.0 * num_corrects/min(batch_size, (len(data) - iter))
        loss.backward()
        optim.step()
        steps+=1

        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {iter+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()    
    
    return total_epoch_loss/steps, total_epoch_acc/steps


# In[8]:


def main(params):
    
    data = Dataset()
    data.load_data('yahoo')
    #from sklearn.utils import shuffle
    #data.train, data.train_lab = shuffle(data.train, data.train_lab)
    data.train = [torch.tensor(x) for x in data.train]
    data.test = [torch.tensor(x) for x in data.test]
    data.val = [torch.tensor(x) for x in data.val]
    data.train_lab = torch.tensor([np.argmax(x) for x in data.train_lab], dtype = torch.int64)
    data.test_lab = torch.tensor([np.argmax(x) for x in data.test_lab], dtype = torch.int64)
    data.val_lab = torch.tensor([np.argmax(x) for x in data.val_lab], dtype = torch.int64)
    
    batch_size = 256
    n_hidden = 300
    input_size = 300 #I guess for n-gram it will be n*300

    W_embd = np.array(cPickle.load(open(data.embpath, 'rb'), encoding = "latin1"))
    W_embd = torch.from_numpy(W_embd)
    classifier = LSTMclassifier(input_size ,n_hidden, data.num_class, W_embd, params["cell"])
   
    num_epoch = 10
    for epoch in range(num_epoch):

        start_time = time.time()        
        train_loss, train_acc = train_model(classifier, data.train, data.train_lab, batch_size, epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rest = divmod(elapsed_time, 3600)
        minutes, sec = divmod(rest, 60)
        """
        Change the path to save the model weights and results
        """
        torch.save(classifier, "./checkpoints/yahoo/lstm1_yahoo_epoch"+str(epoch+1)+".pth")

        val_loss, val_acc = eval_model(classifier, data.val, data.val_lab, batch_size)
        test_loss, test_acc = eval_model(classifier, data.test, data.test_lab, batch_size)
        print(f'Epoch: {epoch+1:02}, Time(hr,min): {hours, minutes},Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%,Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        text_file = open("results_lstm1_yahoo" + ".txt", "a+")
        n = text_file.write(f'Epoch: {epoch+1:02}, Time(hr,min): {hours, minutes},Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%,Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        m = text_file.write("\n")
        text_file.close()
    
    #test_loss, test_acc = eval_model(classifier, data.test, data.test_lab, batch_size)
    #text_file = open("results" + ".txt", "a+")
    #n = text_file.write(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    #m = text_file.write("\n")
    #text_file.close()
    
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    print('done') 


# In[ ]:


if __name__ == '__main__':
    
    d=defaultdict(list)
    """
    pass input as --cell=RKM-LSTM
    """
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
        d[k] = v
    
    main(d)


# In[ ]:




