#!/usr/bin/env python
# coding: utf-8

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


model = Seq2Seq(input_dim=40,vocab_size=len(letter_list),hidden_dim=512,value_size=128, key_size=128)
model = model.to(device)
optimizer = torch.optim.Adam (model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduce=False).to(device)

