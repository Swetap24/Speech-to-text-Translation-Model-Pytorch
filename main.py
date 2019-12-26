#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.utils as utils
from torch.autograd import Variable
from torch.distributions.gumbel import Gumbel
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
import pickle as pk
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# In[3]:


speech_train = np.load('train file path', allow_pickle=True, encoding='bytes')
speech_valid = np.load('validation file path', allow_pickle=True, encoding='bytes')
speech_test = np.load('test file path', allow_pickle=True, encoding='bytes')

transcript_train = np.load('train transcripts file for testing', allow_pickle=True,encoding='bytes')
transcript_valid = np.load('validation transcripts file for testing', allow_pickle=True,encoding='bytes')


# In[ ]:


letter_list = ['&','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


# In[ ]:


a=transcript_train
alls=[]
for i, each in enumerate(a):
    m=[]
    for p, item in enumerate(a[i]):
        for f, item2 in enumerate(a[i][p]):
          if p==0 and f==0:
              m.insert(0, '<sos>')
              m.append(a[i][p].decode()[f])
          elif p==len(a[i])-1 and f==len(a[i][p])-1:
              m.append(a[i][p].decode()[f])
              m.append('<eos>')
          elif f==len(a[i][p])-1:
              m.append(a[i][p].decode()[f])
              m.append(" ")
          else:
              m.append(a[i][p].decode()[f])
    alls.append(m)


# In[ ]:


a=transcript_valid
allsv=[]
for i, each in enumerate(a):
    m=[]
    for p, item in enumerate(a[i]):
        for f, item2 in enumerate(a[i][p]):
          if p==0 and f==0:
              m.insert(0, '<sos>')
              m.append(a[i][p].decode()[f])
          elif p==len(a[i])-1 and f==len(a[i][p])-1:
              m.append(a[i][p].decode()[f])
              m.append('<eos>')
          elif f==len(a[i][p])-1:
              m.append(a[i][p].decode()[f])
              m.append(" ")
          else:
              m.append(a[i][p].decode()[f])
    allsv.append(m)


# In[ ]:


def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list=[]   
    for i, o in enumerate(transcript):
      index_list=[] 
      for s, a in enumerate(transcript[i]):
          check =  all(a in letter_list for item in transcript[i])
          if check is True:
              index_list.append(letter_list.index(a))
      letter_to_index_list.append(index_list)
    return letter_to_index_list


# In[9]:


character_text_train = transform_letter_to_index(alls, letter_list)
character_text_valid = transform_letter_to_index(allsv, letter_list)


# In[ ]:


class pBLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim):
      super(pBLSTM, self).__init__()
      self.blstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True)
  def forward(self,x):
    return self.blstm(x)


# In[ ]:


class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True)
    self.pBLSTM1= pBLSTM(2*hidden_dim, hidden_dim)
    self.pBLSTM2= pBLSTM(2*hidden_dim, hidden_dim)
    self.pBLSTM3= pBLSTM(2*hidden_dim, hidden_dim)
    self.key_network = nn.Linear(hidden_dim*2, value_size)
    self.value_network = nn.Linear(hidden_dim*2, key_size)
  
  def forward(self, x, lens):
        rnn_inp=utils.rnn.pack_padded_sequence(x, lengths=lens, enforce_sorted=False)
        outputs, _=self.lstm(rnn_inp)
        linear_input, _=utils.rnn.pad_packed_sequence(outputs)
        
        for i in range(3):
            if linear_input.shape[0]%2!=0:
                linear_input = linear_input[:-1,:,:]
            outputs = torch.transpose(linear_input, 0, 1)
            outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1]//2, 2, outputs.shape[2])
            outputs = torch.mean(outputs, 2)
            outputs = torch.transpose(outputs,0,1)
            lens=lens//2
            rnn_inp = utils.rnn.pack_padded_sequence(outputs, lengths=lens, enforce_sorted=False)
            if i==0:
                outputs, _=self.pBLSTM1(rnn_inp)
            elif i==1:
                outputs, _=self.pBLSTM2(rnn_inp)
            else:
                outputs, _=self.pBLSTM3(rnn_inp)
            linear_input, _=utils.rnn.pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens


# In[ ]:


class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
  def forward(self, query, key, value, lens):
    '''
    :param query :(N,context_size) Query is the output of LSTMCell from Decoder
    :param key: (T,N,key_size) Key Projection from Encoder per time step
    :param value: (T,N,value_size) Value Projection from Encoder per time step
    :return output: Attended Context
    :return attention_mask: Attention mask that can be plotted  
    '''
    key=torch.transpose(key, 0, 1)
    attention=torch.bmm(key, query.unsqueeze(2)).squeeze(2)
    mask=torch.arange(attention.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
    mask=mask.to(device)
    attention.masked_fill_(mask, -1e9)
    attention=nn.functional.softmax(attention, dim=1)
    value=torch.transpose(value,0,1)
    context=torch.bmm(attention.unsqueeze(1), value).squeeze(1)
    return context, attention


# In[ ]:


class Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128,  isAttended=True):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
    
    self.lstm1 = nn.LSTMCell(input_size=hidden_dim+value_size, hidden_size=hidden_dim)
    self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
    self.isAttended = isAttended
    if(isAttended):
      self.attention = Attention()
    self.character_prob = nn.Linear(key_size+value_size,vocab_size)

  def forward(self, key, values, lens, text=None, train=True):
    '''
    :param key :(T,N,key_size) Output of the Encoder Key projection layer
    :param values: (T,N,value_size) Output of the Encoder Value projection layer
    :param text: (N,text_len) Batch input of text with text_length
    :param train: Train or eval mode
    :return predictions: Returns the character perdiction probability 
    '''
    batch_size=key.shape[1]
    if(train):
      text=torch.transpose(text,0,1)
      max_len=text.shape[1]
      embeddings=self.embedding(text)
    else:
      max_len = 250
    
    predictions = []
    hidden_states = [None, None]
    prediction = torch.zeros(batch_size,1).to(device)
    context=values[0,:,:]
    for i in range(max_len):
      if(train):
          if np.random.random_sample() > 0.6:
              prediction = Gumbel(prediction.to('cpu'), torch.tensor([0.4])).sample().to(device)
              char_embed = self.embedding(prediction.argmax(dim=-1))
          else:
              char_embed = embeddings[:,i,:]
      else:
          char_embed = self.embedding(prediction.argmax(dim=-1))
     
      inp = torch.cat([char_embed,context], dim=1)
      hidden_states[0] = self.lstm1(inp,hidden_states[0])
      
      inp_2 = hidden_states[0][0]
      hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

      output = hidden_states[1][0]
      context, attention=self.attention(output, key, values, lens)
      prediction = self.character_prob(torch.cat([output, context], dim=1))
      predictions.append(prediction.unsqueeze(1))

    return torch.cat(predictions, dim=1)


# In[ ]:


class Seq2Seq(nn.Module):
  def __init__(self,input_dim,vocab_size,hidden_dim,value_size=128, key_size=128,isAttended=True):
    super(Seq2Seq,self).__init__()

    self.encoder = Encoder(input_dim, hidden_dim)
    self.decoder = Decoder(vocab_size, hidden_dim, isAttended=True)
  def forward(self,speech_input, speech_len, text_input=None,train=True):
    key, value, length = self.encoder(speech_input, speech_len)
    if(train):
      predictions = self.decoder(key, value, length, text_input, train=True)
    else:
      predictions = self.decoder(key, value, length, text=None, train=False)
    return predictions


# In[ ]:


class Speech2Text_Dataset(Dataset):
  def __init__(self, speech, text=None, train=True):
    self.speech = speech
    self.train = train
    if(text is not None):
      self.text = text
  def __len__(self):
    return self.speech.shape[0]
  def __getitem__(self, index):
    if(self.train):
      text = self.text[index]
      return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(text[:-1]), torch.tensor(text[1:])
    else:
      return torch.tensor(self.speech[index].astype(np.float32))


# In[ ]:


def collate_train(batch_data):
    inputs, text_input, Labels = zip(*batch_data)
    #Input
    lens=[len(seq) for seq in inputs]
    inputs=[inputs[i] for i in range(len(lens))]
    Len1=torch.LongTensor([len(inp) for inp in inputs])
    inputs=utils.rnn.pad_sequence(inputs)
    #Pred
    lens2=[len(seq) for seq in text_input]
    text_input=[text_input[i] for i in range(len(lens2))]
    Len2=torch.LongTensor([len(inp) for inp in text_input])
    text_input = utils.rnn.pad_sequence(text_input)
    #True
    lens3=[len(seq) for seq in Labels]   
    Labels=[Labels[i] for i in range(len(lens3))]   
    Len3=torch.LongTensor([len(inp) for inp in Labels])
    Labels = utils.rnn.pad_sequence(Labels)
    
    return inputs, text_input, Labels, Len1, Len2, Len3


# In[ ]:


def collate_test(batch_data):
    x=batch_data
    lens=[len(seq) for seq in x]
    x1=sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    x=[x[i] for i in x1]
    len1=torch.LongTensor([len(inp) for inp in x])
    x=utils.rnn.pad_sequence(x)
    return x, len1


# In[ ]:


Speech2Text_train_Dataset = Speech2Text_Dataset(speech_train, character_text_train)
Speech2Text_val_Dataset = Speech2Text_Dataset(speech_valid, character_text_valid)
Speech2Text_test_Dataset = Speech2Text_Dataset(speech_test, train=False)


# In[ ]:


train_loader = DataLoader(Speech2Text_train_Dataset, batch_size=64, shuffle=True, collate_fn=collate_train)
val_loader = DataLoader(Speech2Text_val_Dataset, batch_size=1, shuffle=False, collate_fn=collate_train)
test_loader = DataLoader(Speech2Text_test_Dataset, batch_size=1, shuffle=False, collate_fn=collate_test)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[26]:


model = Seq2Seq(input_dim=40,vocab_size=len(letter_list),hidden_dim=512,value_size=128, key_size=128)
model = model.to(device)
optimizer = torch.optim.Adam (model.parameters(), lr=0.00025)
criterion = nn.CrossEntropyLoss(reduce=False).to(device)


# In[ ]:


def train(model,train_loader, num_epochs, criterion, optimizer):
  for epochs in range(num_epochs):
    model.train()
    loss_sum = 0
    for batch_num,(speech_input, text_input, Labels, speech_len, text_len, Labels_len) in enumerate(train_loader):
      with torch.autograd.set_detect_anomaly(True):
          speech_input=speech_input.to(device)
          text_input=text_input.to(device)
          Labels=Labels.to(device)

          optimizer.zero_grad()
          pred = model(speech_input, speech_len, text_input)
          mask = torch.zeros(Labels.T.size()).to(device)
          pred = pred.contiguous().view(-1, pred.size(-1))
          Labels = torch.transpose(Labels,0,1).contiguous().view(-1)

          for idx, length in enumerate(Labels_len):
              mask[:length,idx] = 1
          
          mask = mask.contiguous().view(-1).to(device)
          loss = criterion(pred, Labels)
          masked_loss = torch.sum(loss*mask)
          masked_loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
          optimizer.step()
          current_loss = float(masked_loss.item())/int(torch.sum(mask).item())

          if  batch_num % 25 == 1:
            print('Epoch: ',epochs, 'Train_loss: ', current_loss)
          torch.cuda.empty_cache()


# In[ ]:


train(model, train_loader, 5, criterion, optimizer)


# In[ ]:


model = torch.save("save path")


# In[ ]:


model = torch.load("load path")

