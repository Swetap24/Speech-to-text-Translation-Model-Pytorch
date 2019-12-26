#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# # Character level data preparation

# In[ ]:


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


# In[ ]:


character_text_train = transform_letter_to_index(alls, letter_list)
character_text_valid = transform_letter_to_index(allsv, letter_list)


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

