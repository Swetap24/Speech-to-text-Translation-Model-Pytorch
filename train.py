#!/usr/bin/env python
# coding: utf-8

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

