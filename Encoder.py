#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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

