import torch
import gensim
import numpy as np
import pickle as cPickle
import torch.optim as optim
import time
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

from collections import defaultdict

from kernel import LSTMcell
from data_load import Dataset
from sklearn.utils import shuffle

import torch.nn as nn
import torch.nn.functional as F

class LSTMclassifier(nn.Module):
    
    """
    Classification task on LSTM output
    """
    
    def __init__(self,input_size, hidden_size, output_size, glove_weights, cell, n_gram):
        
        super(LSTMclassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.labels = output_size
        self.n_gram = n_gram
        
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
            input_seq = input[:,i,:]
            
            """n_gram implementation"""
            for index in range(1, self.n_gram):
                if i-index < 0:
                    input[:,i-index,:] = torch.zeros(batch_size, 300)
                    if torch.cuda.is_available():
                        input[:,i-index,:] = input[:,i-index,:].cuda()
                input_seq = torch.cat((input_seq, input[:,i-index,:]), 1)
            
            """"""
            hidden_state, cell_state = self.lstm(input_seq, hidden_state, cell_state)
            output[:,i,:] = hidden_state
        
        pool = nn.AvgPool2d((max_num_of_words,1), stride=1)
        time_avg_output = torch.squeeze(pool(output))
        linear_layer = torch.sigmoid(self.layer1(time_avg_output))
        final_output = torch.sigmoid(self.layer2(linear_layer))
        final_output = self.softmax(final_output)
        
        return final_output

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


def main(params):
    
    data = Dataset()
    data.load_data(params["data"])
    data.train, data.train_lab = shuffle(data.train, data.train_lab)
    data.train = [torch.tensor(x) for x in data.train]
    data.test = [torch.tensor(x) for x in data.test]
    data.val = [torch.tensor(x) for x in data.val]
    data.train_lab = torch.tensor([np.argmax(x) for x in data.train_lab], dtype = torch.int64)
    data.test_lab = torch.tensor([np.argmax(x) for x in data.test_lab], dtype = torch.int64)
    data.val_lab = torch.tensor([np.argmax(x) for x in data.val_lab], dtype = torch.int64)
    
    batch_size = 256
    n_gram = 1
    num_epoch = 20
    if "num_epoch" in params:
        num_epoch = int(params["num_epoch"])
    if "n_gram" in params:
        n_gram = int(params["n_gram"])
    if "batch" in params:
        batch_size = int(params["batch"])
    
    print("struture used:", params["cell"])
    print("batch_size:", batch_size)
    print("no. epochs:", num_epoch)
    print("n-gram:", n_gram)
    
    n_hidden = 300
    input_size = 300*n_gram 

    W_embd = np.array(cPickle.load(open(data.embpath, 'rb'), encoding = "latin1"))
    W_embd = torch.from_numpy(W_embd)
    classifier = LSTMclassifier(input_size ,n_hidden, data.num_class, W_embd, params["cell"], n_gram)
   
    
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
        torch.save(classifier, "./checkpoints/"+data.data+"_" + params['cell'] +"_epoch"+str(epoch+1)+".pth")

        val_loss, val_acc = eval_model(classifier, data.val, data.val_lab, batch_size)
        test_loss, test_acc = eval_model(classifier, data.test, data.test_lab, batch_size)
        
        print(f'Epoch: {epoch+1:02}, Time(hr,min): {hours, minutes},Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%,Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        
        text_file = open("./result/"+ params['cell'] +"_"+data.data + ".txt", "a+")
        n = text_file.write(f'Epoch: {epoch+1:02}, Time(hr,min): {hours, minutes},Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%,Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        m = text_file.write("\n")
        text_file.close()
    
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    print('done') 


if __name__ == '__main__':
    
    d=defaultdict(list)
    """
    pass input as 
    --cell = RKM-LSTM
    --data = agnews
    --batch = 64 (default value is 256)
    --n_gram = 3 (default value is 1)
    --num_epoch = 64 (default value is 256)
    """
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
        d[k] = v
    
    main(d)




